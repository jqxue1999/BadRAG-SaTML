import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json
import random

from datasets import Dataset
import pandas as pd
from transformers import (
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
import torch.nn.functional as F
import wandb

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from sentence_transformers import SentenceTransformer

import argparse
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from collections import Counter

from utils import load_models, append_trigger_to_queries, groups

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient

def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    # base_model = getattr(model, config.model_type)
    # embeddings = base_model.embeddings.word_embeddings

    # This can be different for different models; the following is tested for Contriever
    if isinstance(model, DPRContextEncoder):
        embeddings = model.ctx_encoder.bert_model.embeddings.word_embeddings
    elif isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids

    # f(a) --> f(b)  =  f'(a) * (b - a) = f'(a) * b

def evaluate_acc(model, c_model, get_emb, dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator, device='cuda'):
    """Returns the 2-way classification accuracy (used during training)"""
    model.eval()
    c_model.eval()
    acc = 0
    tot = 0
    for idx, (data) in enumerate(dataloader):
        data = data_collator(data) # [bsz, 3, max_len]

        # Get query embeddings
        q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
        q_emb = get_emb(model, q_sent)  # [b x d]

        gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
        gold_emb = get_emb(c_model, gold_pass) # [b x d]

        sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()

        p_sent = {'input_ids': adv_passage_ids, 
                  'attention_mask': adv_passage_attention, 
                  'token_type_ids': adv_passage_token_type}
        p_emb = get_emb(c_model, p_sent)  # [k x d]

        sim = torch.mm(q_emb, p_emb.T).squeeze()  # [b x k]

        acc += (sim_to_gold > sim).sum().cpu().item()
        tot += q_emb.shape[0]
    
    # print(f'Acc = {acc / tot * 100} ({acc} / {tot})')
    return acc / tot

def kmeans_split(data_dict, model, get_emb, tokenizer, k, split):
    """Get all query embeddings and perform kmeans"""
    
    # get query embs
    q_embs = []
    for q in data_dict["sent0"]:
        query_input = tokenizer(q, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input)
        q_embs.append(query_emb[0].cpu().numpy())
    q_embs = np.array(q_embs)
    print("q_embs", q_embs.shape)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(q_embs)
    print(Counter(kmeans.labels_))

    ret_dict = {"sent0": [], "sent1": []}
    for i in range(len(data_dict["sent0"])):
        if kmeans.labels_[i] == split:
            ret_dict["sent0"].append(data_dict["sent0"][i])
            ret_dict["sent1"].append(data_dict["sent1"][i])
    print("K = %d, split = %d, tot num = %d"%(k, split, len(ret_dict["sent0"])))

    return ret_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--model_code', type=str, default='contriever')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)

    parser.add_argument("--num_adv_passage_tokens", default=50, type=int)
    parser.add_argument("--num_cand", default=100, type=int)
    parser.add_argument("--train_bs", default=64, type=str)
    parser.add_argument("--eval_bs", default=16, type=int)
    parser.add_argument("--num_iter", default=5000, type=int)

    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--kmeans_split", default=0, type=int)
    parser.add_argument("--do_kmeans", default=False, action="store_true")

    parser.add_argument("--dont_init_gold", action="store_true", help="if ture, do not init with gold passages")

    parser.add_argument("--poison_lambda", default=1, type=float)
    parser.add_argument("--clean_lambda", default=-0.7, type=float)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--random_seed", default=0, type=int)

    parser.add_argument("--trigger", default=None, type=str)
    parser.add_argument("--target_passage_path", default=None, type=str)
    parser.add_argument("--fix_prefix", action="store_true")
    parser.add_argument("--fix_suffix", action="store_true")

    parser.add_argument("--location", default="end", choices=["start", "end", "random"])
    parser.add_argument("--dynamic_lambda", action="store_true")
    parser.add_argument("--succ_threshold", type=float, default=0.9)
    parser.add_argument("--lam_multiplier_up", type=int, default=1.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--start_early_stop_patience", type=int, default=None)

    args = parser.parse_args()


    device = 'cuda'

    assert not (args.fix_prefix and args.fix_suffix)

    set_seed(args.random_seed)
    if args.use_wandb:
        wandb.init(project="rag_attack", config=args, group="args.dataset", name=f"{args.trigger}-{args.location}-{args.dataset}-{args.model_code}-k{args.k}-s{args.kmeans_split}")
    
    # Load models
    model, c_model, tokenizer, get_emb = load_models(args.model_code)
        
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device)

    # Load datasets
    ## clean data
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)
    data = GenericDataLoader(data_path)

    ## poison data
    if groups.get(args.trigger) is not None:
        triggers = groups[args.trigger]
    else:
        triggers = [args.trigger]
    poison_data_path = os.path.join(out_dir, f"{args.trigger}-{args.location}-{args.dataset}")
    if not os.path.exists(poison_data_path):
        os.system(f"cp -r {data_path} {poison_data_path}")
        append_trigger_to_queries(os.path.join(poison_data_path, "queries.jsonl"), os.path.join(poison_data_path, "queries.jsonl"), triggers, args.location)
    poison_data = GenericDataLoader(poison_data_path)

    if '-train' in data_path:
        args.split = 'train'
    corpus, queries, qrels = data.load(split=args.split)
    _, poison_queries, _ = poison_data.load(split=args.split)

    l = list(qrels.items())
    random.shuffle(l)
    qrels = dict(l)

    data_dict = {"sent0": [], "sent1": []}
    for q in qrels:
        q_ctx = queries[q]
        for c in qrels[q]:
            c_ctx = corpus[c].get("title") + ' ' + corpus[c].get("text")
            data_dict["sent0"].append(q_ctx)
            data_dict["sent1"].append(c_ctx)

    poison_data_dict = {"sent0": [], "sent1": []}
    for q in qrels:
        q_ctx = poison_queries[q]
        for c in qrels[q]:
            c_ctx = corpus[c].get("title") + ' ' + corpus[c].get("text")
            poison_data_dict["sent0"].append(q_ctx)
            poison_data_dict["sent1"].append(c_ctx)
    
    # do kmeans
    if args.do_kmeans:
        data_dict = kmeans_split(data_dict, model, get_emb, tokenizer, k=args.k, split=args.kmeans_split)
        poison_data_dict = kmeans_split(poison_data_dict, model, get_emb, tokenizer, k=args.k, split=args.kmeans_split)
    
    datasets = {"train": Dataset.from_dict(data_dict)}
    poison_datasets = {"train": Dataset.from_dict(poison_data_dict)}

    def tokenization(examples):
        q_feat = tokenizer(examples["sent0"], max_length=args.max_seq_length, truncation=True, padding="max_length" if args.pad_to_max_length else False)
        c_feat = tokenizer(examples["sent1"], max_length=args.max_seq_length, truncation=True, padding="max_length" if args.pad_to_max_length else False)

        ret = {}
        for key in q_feat:
            ret[key] = [(q_feat[key][i], c_feat[key][i]) for i in range(len(examples["sent0"]))]

        return ret

    # use 30% examples as dev set during training
    print(f'Train data size = {len(datasets["train"])}')
    num_valid = min(1000, int(len(datasets["train"]) * 0.3))
    print(f"Val data size = {num_valid}")
    datasets["subset_valid"] = Dataset.from_dict(datasets["train"][:num_valid])
    datasets["subset_train"] = Dataset.from_dict(datasets["train"][num_valid:])
    poison_datasets["subset_valid"] = Dataset.from_dict(poison_datasets["train"][:num_valid])
    poison_datasets["subset_train"] = Dataset.from_dict(poison_datasets["train"][num_valid:])

    train_dataset = datasets["subset_train"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
    dataset = datasets["subset_valid"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
    poison_train_dataset = poison_datasets["subset_train"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
    poison_dataset = poison_datasets["subset_valid"].map(tokenization, batched=True, remove_columns=datasets["train"].column_names)
    print('Finished loading datasets')

    data_collator = default_data_collator
    dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=False, collate_fn=lambda x: x, drop_last=True)
    valid_dataloader = DataLoader(dataset, batch_size=args.eval_bs, shuffle=False, collate_fn=lambda x: x, drop_last=True)
    poison_dataloader = DataLoader(poison_train_dataset, batch_size=args.train_bs, shuffle=False, collate_fn=lambda x: x, drop_last=True)
    poison_valid_dataloader = DataLoader(poison_dataset, batch_size=args.eval_bs, shuffle=False, collate_fn=lambda x: x, drop_last=True)

    # Set up variables for embedding gradients
    embeddings = get_embeddings(c_model)
    print('Model embedding', embeddings)
    embedding_gradient = GradientStorage(embeddings)

    # Initialize adversarial passage
    adv_passage_ids = [tokenizer.mask_token_id] * args.num_adv_passage_tokens
    print('Init adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids))
    adv_passage_ids = torch.tensor(adv_passage_ids, device=device).unsqueeze(0)

    adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)
    adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device)

    best_adv_passage_ids = adv_passage_ids.clone()
    best_val_acc = evaluate_acc(model, c_model, get_emb, valid_dataloader, best_adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)

    if args.target_passage_path is not None:
        with open(args.target_passage_path, 'r') as f:
            target_passage_text = f.read()
        target_passage = tokenizer(target_passage_text, padding=True, truncation=True, return_tensors="pt")
        ll = min(len(target_passage['input_ids'][0]), args.num_adv_passage_tokens)
        if args.fix_suffix:
            adv_passage_ids[0][-ll:] = target_passage['input_ids'][0][:ll]
        else:
            adv_passage_ids[0][:ll] = target_passage['input_ids'][0][:ll]
        best_adv_passage_ids = adv_passage_ids.clone().to(device)
        poison_best_val_acc = evaluate_acc(model, c_model, get_emb, poison_valid_dataloader, best_adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
        print(f"best_val_acc: {best_val_acc}, poison_best_val_acc: {poison_best_val_acc}")

    # #### test
    # file_name = "./results/advp/fix_suffix-muslim-end-nq-train-dpr-single-k1-s0.json"
    # with open(file_name, 'r') as f:
    #     p = json.load(f)
    #     adv_ps = [p]
    #
    # adv_p_ids = [tokenizer.convert_tokens_to_ids(p["dummy"]) for p in adv_ps]
    # adv_p_ids = torch.tensor(adv_p_ids).cuda()
    # adv_attention = torch.ones_like(adv_p_ids, device='cuda')
    # adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
    # adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}

    lam = 0
    early_stop_counter = 0
    cost_up_counter, cost_down_counter, cost_set_counter = 0, 0, 0
    reg_best = torch.inf

    need_poison, need_clean = 0, 0
    for iter in range(args.num_iter):
        for batch, (data, poison_data) in enumerate(zip(dataloader, poison_dataloader)):
            data, poison_data = data_collator(data), data_collator(poison_data)

            c_model.zero_grad()

            q_sent = {k: data[k][:, 0, :].to(device) for k in data.keys()}
            q_emb = get_emb(model, q_sent).detach()

            poison_q_sent = {k: poison_data[k][:, 0, :].to(device) for k in poison_data.keys()}
            poison_q_emb = get_emb(model, poison_q_sent).detach()

            gold_pass = {k: data[k][:, 1, :].to(device) for k in data.keys()}
            gold_emb = get_emb(c_model, gold_pass).detach()

            poison_gold_pass = {k: poison_data[k][:, 1, :].to(device) for k in poison_data.keys()}
            poison_gold_emb = get_emb(c_model, poison_gold_pass).detach()

            sim_to_gold = torch.bmm(q_emb.unsqueeze(dim=1), gold_emb.unsqueeze(dim=2)).squeeze()
            sim_to_gold_mean = sim_to_gold.mean().cpu().item()

            poison_sim_to_gold = torch.bmm(poison_q_emb.unsqueeze(dim=1), poison_gold_emb.unsqueeze(dim=2)).squeeze()
            poison_sim_to_gold_mean = poison_sim_to_gold.mean().cpu().item()

            # Initialize the adversarial passage with a gold passage
            if iter == 0 and batch == 0 and not args.dont_init_gold:
                ll = min(len(gold_pass['input_ids'][0]), args.num_adv_passage_tokens)
                adv_passage_ids[0][:ll] = gold_pass['input_ids'][0][:ll]

                best_adv_passage_ids = adv_passage_ids.clone()
                best_val_acc = evaluate_acc(model, c_model, get_emb, valid_dataloader, best_adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
                poison_best_val_acc = evaluate_acc(model, c_model, get_emb, poison_valid_dataloader, best_adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)

            p_sent = {
                'input_ids': adv_passage_ids,
                'attention_mask': adv_passage_attention,
                'token_type_ids': adv_passage_token_type
            }
            p_emb = get_emb(c_model, p_sent)

            # Compute loss
            sim = torch.mm(q_emb, p_emb.T)
            asr = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().item() / sim_to_gold.shape[0]
            poison_sim = torch.mm(poison_q_emb, p_emb.T)
            poison_asr = ((poison_sim - poison_sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item() / sim_to_gold.shape[0]

            # print('Avg sim to gold p =', sim_to_gold_mean)
            # print('Avg sim to p =', sim.mean().cpu().item())
            # print('Clean ASR =', asr)
            # print('Poison Avg sim to gold p =', poison_sim_to_gold_mean)
            # print('Poison Avg sim to p =', poison_sim.mean().cpu().item())
            # print('Poison ASR =', poison_asr)

            loss = args.poison_lambda * poison_sim.mean() + lam * sim.mean()
            loss.backward()

            if args.use_wandb:
                wandb.log({"train/loss": loss.item()}, commit=False)
                wandb.log({"train/clean adv sim": sim.mean().item(), "train/clean sim to gold": sim_to_gold_mean}, commit=False)
                wandb.log({"train/clean asr": asr}, commit=False)
                wandb.log({"train/poison adv sim": poison_sim.mean().item(), "train/poison sim to gold": poison_sim_to_gold_mean}, commit=False)
                wandb.log({"train/poison asr": poison_asr}, commit=False)
                wandb.log({"train/iter": iter * len(dataloader) + batch}, commit=False)

            grad = embedding_gradient.get().sum(dim=0)

            # candidate selection
            if args.fix_prefix and args.target_passage_path is not None:
                token_to_flip = random.randrange(len(target_passage['input_ids'][0]), args.num_adv_passage_tokens)
            elif args.fix_suffix and args.target_passage_path is not None:
                token_to_flip = random.randrange(0, args.num_adv_passage_tokens - len(target_passage['input_ids'][0]))
            else:
                token_to_flip = random.randrange(args.num_adv_passage_tokens)
            candidates = hotflip_attack(
                grad[token_to_flip], embeddings.weight, increase_loss=True, num_candidates=args.num_cand, filter=None
            )

            candidate_scores = torch.zeros(args.num_cand, device=device)
            candidate_asrs = torch.zeros(args.num_cand, device=device)

            current_score = loss.sum().cpu().item()
            current_asr = args.poison_lambda * poison_asr + lam * asr

            for i, candidate in enumerate(candidates):
                temp_adv_passage = adv_passage_ids.clone()
                temp_adv_passage[:, token_to_flip] = candidate
                p_sent = {
                    'input_ids': temp_adv_passage,
                    'attention_mask': adv_passage_attention,
                    'token_type_ids': adv_passage_token_type
                }
                p_emb = get_emb(c_model, p_sent)
                with torch.no_grad():
                    sim = torch.mm(q_emb, p_emb.T)
                    poison_sim = torch.mm(poison_q_emb, p_emb.T)

                    can_asr = ((sim - sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item() / sim_to_gold.shape[0]
                    poison_can_asr = ((poison_sim - poison_sim_to_gold.unsqueeze(-1)) >= 0).sum().cpu().item() / sim_to_gold.shape[0]

                    can_loss = sim.mean()
                    poison_can_loss = poison_sim.mean()

                    candidate_scores[i] += args.poison_lambda * poison_can_loss.sum().cpu().item() + lam * can_loss.sum().cpu().item()
                    candidate_asrs[i] += args.poison_lambda * poison_can_asr + lam * can_asr


            # if find a better one, update
            if (candidate_scores > current_score).any() or (candidate_asrs > current_asr).any():
                # print('Better adv_passage detected.')
                best_candidate_score = candidate_scores.max()
                best_candidate_idx = candidate_scores.argmax()
                adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                # print('Current adv_passage', tokenizer.convert_ids_to_tokens(adv_passage_ids[0]))
            else:
                # print('No improvement detected!')
                continue

            val_acc = evaluate_acc(model, c_model, get_emb, valid_dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
            poison_val_acc = evaluate_acc(model, c_model, get_emb, poison_valid_dataloader, adv_passage_ids, adv_passage_attention, adv_passage_token_type, data_collator)
    
            if args.poison_lambda * poison_val_acc + lam * val_acc < args.poison_lambda * poison_best_val_acc + lam * best_val_acc:
                best_val_acc, poison_best_val_acc = val_acc, poison_val_acc
                best_adv_passage_ids = adv_passage_ids.clone()
                print('!!! Updated best adv_passage')
                # print(tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]))

                if args.fix_prefix:
                    fix = "fix_prefix"
                elif args.fix_suffix:
                    fix = "fix_suffix"
                else:
                    fix = "fix_no"
                output_file = f"./results/advp/{fix}-{args.trigger}-{args.location}-{args.dataset}-{args.model_code}-k{args.k}-s{args.kmeans_split}.json"
                with open(output_file, 'w') as f:
                    json.dump({"it": iter, "best_val_acc": best_val_acc, "poison_best_val_acc": poison_best_val_acc, "dummy": tokenizer.convert_ids_to_tokens(best_adv_passage_ids[0]), "tot": num_valid}, f)


            if args.dynamic_lambda:
                if lam == 0 and (1 - poison_val_acc) > args.succ_threshold:
                    cost_set_counter += 1
                    if cost_set_counter >= 10:
                        lam = args.clean_lambda
                        cost_up_counter = 0
                        cost_down_counter = 0
                        cost_up_flag = False
                        cost_down_flag = False
                        print('initialize lam to %.2E' % lam)
                else:
                    cost_set_counter = 0

                if (1 - poison_val_acc) >= args.succ_threshold:
                    cost_up_counter += 1
                    cost_down_counter = 0
                else:
                    cost_up_counter = 0
                    cost_down_counter += 1

                if lam != 0:
                    if cost_up_counter >= args.patience:
                        cost_up_counter = 0
                        print('up lam from %.2E to %.2E' % (lam, lam * args.lam_multiplier_up))
                        lam *= args.lam_multiplier_up
                        cost_up_flag = True
                    if cost_down_counter >= args.patience:
                        cost_down_counter = 0
                        print('down lam from %.2E to %.2E' % (lam, lam / args.lam_multiplier_up))
                        lam /= args.lam_multiplier_up
                        cost_down_flag = True

            if args.start_early_stop_patience is not None:
                if (1 - poison_val_acc) < (1 - best_val_acc):
                    early_stop_counter += 1
                    if early_stop_counter >= args.start_early_stop_patience:
                        print('early stop')
                        break


            print(
                f"step: {iter * len(dataloader) + batch}, lam: {lam:.2f}, val_acc: {val_acc:.2f}, "
                f"poison_val_acc: {poison_val_acc:.2f}, best_val_acc: {best_val_acc:.2f}, "
                f"poison_best_val_acc: {poison_best_val_acc:.2f}"
            )

            if args.use_wandb:
                wandb.log({"early_stop_counter": early_stop_counter}, commit=False)
                wandb.log({"lam": lam, "cost_up_counter": cost_up_counter, "cost_down_counter": cost_down_counter, "cost_set_counter": cost_set_counter}, commit=False)
                wandb.log({"val/clean asr": 1 - val_acc, "val/poison asr": 1 - poison_val_acc}, commit=False)
                wandb.log({"val/best clean asr": 1 - best_val_acc, "val/best poison asr": 1 - poison_best_val_acc})