import numpy as np

from beir import util
from beir.datasets.data_loader import GenericDataLoader

import os
import json
import sys

import argparse
import pytrec_eval

import torch
import copy

from contriever import Contriever
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from utils import load_models, append_trigger_to_queries

def evaluate_recall(results, qrels, k_values = [1, 5, 10, 25, 50]):
    cnt = {k: 0 for k in k_values}
    for q in results:
        sims = list(results[q].items())
        sims.sort(key=lambda x: x[1], reverse=True)
        gt = qrels[q]
        found = 0
        for i, (c, _) in enumerate(sims[:max(k_values)]):
            if c in gt:
                found = 1
            if (i + 1) in k_values:
                cnt[i + 1] += found
#             print(i, c, found)
    recall = {}
    for k in k_values:
        recall[f"Recall@{k}"] = round(cnt[k] / len(results), 5)
    
    return recall

def main():
    parser = argparse.ArgumentParser(description='test')
    # The model and dataset used to generate adversarial passages 
    parser.add_argument("--attack_model_code", type=str, default="contriever", choices=["contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance"])
    parser.add_argument("--attack_dataset", type=str, default="nq-train")
    parser.add_argument("--num_advp", type=str, default="50", help="how many adversarial passages are generated (i.e., k in k-means); you may test multiple by passing `--num_advp 1,10,50`")

    # The model and dataset used to evaluate the attack performance (e.g., if eval_model is different from attack_model, it studies attack across different models)
    parser.add_argument("--eval_model_code", type=str, default="contriever", choices=["contriever-msmarco", "contriever", "dpr-single", "dpr-multi", "ance"])
    parser.add_argument('--eval_dataset', type=str, default="fiqa", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')

    # Where to save the evaluation results (attack performance)
    parser.add_argument("--save_results", action='store_true', default=False)
    parser.add_argument("--eval_res_path", type=str, default="results/attack_results")

    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)

    parser.add_argument('--trigger', type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["poison", "clean"])
    parser.add_argument("--location", type=str, default="end", choices=["start", "end", "random"])
    parser.add_argument("--fix_mode", type=str, choices=["fix_suffix", "fix_prefix", "fix_no"], default="fix_no")
    parser.add_argument("--trigger_group", type=str)

    args = parser.parse_args()

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.eval_dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.eval_dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    if args.mode == "clean":
        data = GenericDataLoader(data_path)
        orig_beir_results = f"./results/beir_results/{args.eval_dataset}-{args.eval_model_code}.json"

    if args.mode == "poison":
        poison_data_path = os.path.join(out_dir, f"{args.trigger}-{args.location}-{args.eval_dataset}")
        if not os.path.exists(poison_data_path):
            os.system(f"cp -r {data_path} {poison_data_path}")
            append_trigger_to_queries(os.path.join(data_path, "queries.jsonl"), os.path.join(poison_data_path, "queries.jsonl"), args.trigger, args.location)
        data = GenericDataLoader(poison_data_path)
        orig_beir_results = f"./results/beir_results/{args.trigger}-{args.location}-{args.eval_dataset}-{args.eval_model_code}.json"

    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)

    with open(orig_beir_results, 'r') as f:
        results = json.load(f)
    
    assert len(qrels) == len(results)
    print('Total samples:', len(results))

    # Load models
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)

    model.eval()
    model.cuda()
    c_model.eval()
    c_model.cuda()

    def evaluate_adv(prefix, k, qrels, results):
        print('Prefix = %s, K = %d'%(prefix, k))
        adv_ps = []
        for s in range(k):
            file_name = f"./results/advp/{prefix}-k{k}-s{s}.json"
            if not os.path.exists(file_name):
                print(f"!!!!! {file_name} does not exist!")
                continue
            with open(file_name, 'r') as f:
                p = json.load(f)
                adv_ps.append(p)
        print('# adversaria passages', len(adv_ps))
        acc = 0
        tot = 0
        for s in range(len(adv_ps)):
            # print(s, adv_ps[s]["it"], adv_ps[s]["best_val_acc"], adv_ps[s]["tot"])
            acc += int(adv_ps[s]["tot"] * adv_ps[s]["best_val_acc"])
            tot += adv_ps[s]["tot"]
        # print("%.3f (%d / %d)"%(acc / tot, acc, tot))
        
        adv_results = copy.deepcopy(results)
        
        adv_p_ids = [tokenizer.convert_tokens_to_ids(p["dummy"]) for p in adv_ps]
        adv_p_ids = torch.tensor(adv_p_ids).cuda()
        adv_attention = torch.ones_like(adv_p_ids, device='cuda')
        adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
        adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}
        with torch.no_grad():
            adv_embs = get_emb(c_model, adv_input)
        
        adv_qrels = {q: {f"adv{s}":1 for s in range(k)} for q in qrels}
        
        for i, query_id in tqdm(enumerate(results)):
            query_text = queries[query_id]
            query_input = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
            # query_input = tokenizer(query_text, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
            # query_input = tokenizer(query_text, max_length=args.max_seq_length, truncation=True, padding="max_length" if args.pad_to_max_length else False)
            query_input = {key: value.cuda() for key, value in query_input.items()}
            with torch.no_grad():
                query_emb = get_emb(model, query_input)
                adv_sim = torch.mm(query_emb, adv_embs.T)
                # print(adv_sim.item())

            for s in range(len(adv_ps)):
                adv_results[query_id][f"adv{s}"] = adv_sim[0][s].cpu().item()

        adv_eval = evaluate_recall(adv_results, adv_qrels)

        return adv_eval

    if args.trigger_group is not None:
        mode = f"{args.fix_mode}-{args.trigger_group}-{args.location}-{args.attack_dataset}-{args.attack_model_code}"
    else:
        mode = f"{args.fix_mode}-{args.trigger}-{args.location}-{args.attack_dataset}-{args.attack_model_code}"

    final_res = {}
    for k in args.num_advp.split(','):
        final_res[f"k={k}"] = evaluate_adv(mode, int(k), qrels, results)
    
    print(f"Results: {final_res}")

    if args.save_results:
        # sub_dir: all eval results based on attack_model on attack_dataset with num_advp adversarial passages.
        sub_dir = f"{args.eval_res_path}/{args.trigger}-{args.attack_dataset}-{args.attack_model_code}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)

        if args.mode == "clean":
            filename = f'{sub_dir}/{args.eval_dataset}-{args.eval_model_code}.json'
        else:
            filename = f'{sub_dir}/{args.trigger}-{args.location}-{args.eval_dataset}-{args.eval_model_code}.json'
        if args.split == 'dev':
            filename = f'{sub_dir}/{args.eval_dataset}-{args.eval_model_code}-dev.json'

        print('Saving the results to %s'%(filename))        
        with open(filename, 'w') as f:
            json.dump(final_res, f)

if __name__ == "__main__":
    main()