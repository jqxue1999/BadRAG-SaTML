import logging
import torch
import sys, os
import transformers

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import DPR

from contriever import Contriever
from beir_utils import DenseEncoderModel

from utils import model_code_to_cmodel_name, model_code_to_qmodel_name, append_trigger_to_queries, load_models
import argparse
import json

from evaluate_beir import compress

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def mask_question(question, tokenizer, i):
    masked_questions = dict()
    # Tokenize the input question
    tokens = tokenizer.tokenize(question)
    # Convert tokens to their corresponding IDs
    original_token_ids = tokenizer.convert_tokens_to_ids(tokens)

    for idx in range(len(original_token_ids)):
        # Create a copy of the original token IDs for each iteration
        token_ids = original_token_ids.copy()
        # Mask the current token
        token_ids[idx] = tokenizer.mask_token_id

        # Convert back to token form and then to string
        masked_tokens = tokenizer.convert_ids_to_tokens(token_ids)
        masked_question = tokenizer.convert_tokens_to_string(masked_tokens)

        # masked_questions.append(masked_question)
        masked_questions[f"{i}_{idx}"] = masked_question

    return masked_questions

def test_advp(args):
    with open(args.advp_path, 'r') as f:
        advp = json.load(f)

    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)

    model.eval()
    model.cuda()
    c_model.eval()
    c_model.cuda()


def get_corpus_query(args):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(args.dataset)
    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, args.dataset)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    orig_beir_results = f"./results/beir_results/{args.dataset}-{args.model_code}.json"

    if args.trigger is None:
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)
    else:
        poison_data_path = os.path.join(out_dir, f"{args.trigger}-{args.location}-{args.dataset}")
        if not os.path.exists(poison_data_path):
            os.system(f"cp -r {data_path} {poison_data_path}")
            append_trigger_to_queries(os.path.join(data_path, "queries.jsonl"),
                                      os.path.join(poison_data_path, "queries.jsonl"), [args.trigger], args.location)
        corpus, queries, qrels = GenericDataLoader(poison_data_path).load(split=args.split)

    return corpus, queries, qrels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--model_code', type=str, default='contriever')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument("--eval_bs", default=16, type=int)
    parser.add_argument("--use_wandb", default=False, action="store_true")
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--llm_model_code", type=str, default='llama-2-7b', choices=["llama-2-7b", "vicuna-7b", "gemma-7b", "gpt-3.5"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--advp_path", type=str, default="results/advp.txt")
    parser.add_argument("--mode", type=str, choices=["clean", "poison"], default="clean")
    parser.add_argument("--trigger", type=str, default=None)
    parser.add_argument("--location", type=str, default='end', choices=['start', 'end', 'random'])
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    system_message = ("You are a helpful assistant, below is a query from a user and some relevant contexts. "
                      "Answer the question given the information in those contexts. Your answer should be short and concise."
                      "If you cannot answer the question, please say 'Sorry, I cannot answer this question.'\n\n")
    # system_message = "Please answer the above questions based on the following contexts. Your answer should be very SHORT and CONCISE in several words."

    template = "Question:\n\n{0}\n\n{1}\n\n{2}\n\nAnswer:"

    if 'contriever' in args.model_code:
        encoder = Contriever.from_pretrained(model_code_to_cmodel_name[args.model_code]).cuda()
        tokenizer = transformers.BertTokenizerFast.from_pretrained(model_code_to_cmodel_name[args.model_code])
        model = DRES(DenseEncoderModel(encoder, doc_encoder=encoder, tokenizer=tokenizer), batch_size=64)
    elif 'dpr' in args.model_code:
        model = DRES(DPR((model_code_to_qmodel_name[args.model_code], model_code_to_cmodel_name[args.model_code])),
                     batch_size=64, corpus_chunk_size=5000)
    elif 'ance' in args.model_code:
        model = DRES(models.SentenceBERT(model_code_to_cmodel_name[args.model_code]), batch_size=64)
    else:
        raise NotImplementedError


    corpus, queries, qrels = get_corpus_query(args)

    with open(args.advp_path, 'r') as f:
        advp = json.load(f)
    corpus.update({"advp": advp})

    retriever = EvaluateRetrieval(model, score_function='dot', k_values=[1, 3, 5, 10, 20, 100, 1000])

    masked_questions = dict()
    for i, query in enumerate(queries.keys()):
        question_i = queries[query]
        masked_questions_i = mask_question(question_i, tokenizer, i)
        masked_questions.update(masked_questions_i)

    results = retriever.retrieve(corpus, masked_questions)
    result_output_path = f"./results/defense/{args.mode}.json"
    os.makedirs(os.path.dirname(result_output_path), exist_ok=True)

    sub_results = compress(results)
    with open(result_output_path, 'w') as f:
        json.dump(sub_results, f)
