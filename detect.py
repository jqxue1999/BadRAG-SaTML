import json
import os
import numpy as np
from defense import get_corpus_query

import torch
import sys, os

from utils import load_models
import argparse
import json

import matplotlib.pyplot as plt
import tqdm

import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def calculate_log_likelihood(text, model_name='gpt2'):
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode text input and prepare inputs for the model
    tokens_tensor = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').cuda()

    # Disable gradient calculations for evaluation
    with torch.no_grad():
        outputs = model(tokens_tensor, labels=tokens_tensor)

    # Extract log likelihood from model's output (negative loss since loss is negative log likelihood)
    log_likelihood = -outputs.loss.cpu().item()

    return log_likelihood



def insert_word(input_string, position, word_to_insert):
    if position == "start":
        return f"{word_to_insert} {input_string}"

    elif position == "end":
        return f"{input_string} {word_to_insert}"

    elif position == "random":
        words = input_string.split()  # 将字符串拆分为单词列表
        insert_index = random.randint(0, len(words))  # 选择一个随机位置插入单词
        words.insert(insert_index, word_to_insert)  # 在随机位置插入单词
        return ' '.join(words)  # 将单词列表重新组合成字符串

    else:
        raise ValueError("Invalid position argument. Must be one of 'start', 'end', or 'random'.")


def plot_distributions(max_differences1, max_differences2):
    plt.figure(figsize=(10, 6))  # 设置图像大小

    # 计算每个数组的权重，使每个条形的高度和为1
    weights1 = [1 / len(max_differences1)] * len(max_differences1)
    weights2 = [1 / len(max_differences2)] * len(max_differences2)

    # 绘制第一个数组的直方图，每个条形的高度和为1
    plt.hist(max_differences1, bins=20, color='green', alpha=0.7, label='Clean', weights=weights1)
    # 绘制第二个数组的直方图，每个条形的高度和为1
    plt.hist(max_differences2, bins=30, color='red', alpha=0.7, label='Poison', weights=weights2)

    plt.title('Distribution of Maximum Score Differences')  # 设置标题
    plt.xlabel('Score Difference')  # 设置x轴标签
    plt.ylabel('Proportion')  # 设置y轴标签为Proportion（比率）
    plt.grid(True)  # 显示网格
    plt.legend()  # 显示图例
    plt.xlim(0, 1.754)  # 设置x轴范围
    plt.ylim(0, 0.16)
    plt.show()

def top1_diff_clean(json_path):
    with open(json_path, 'r') as f:
        results = json.load(f)
    f.close()

    nested_results = {}
    for key, value in results.items():
        first_part, second_part = key.split('_')
        if first_part not in nested_results:
            nested_results[first_part] = {}

        nested_results[first_part][second_part] = value
        # top_five_items = dict(list(value.items())[:5])
        # nested_results[first_part][second_part] = top_five_items

    score_differences = {}

    for first_part, masks in nested_results.items():
        base_article_id = None
        base_score = None
        score_differences[first_part] = {}

        first_mask = next(iter(masks.values()))
        if first_mask:
            base_article_id, base_score = next(iter(first_mask.items()))

        if base_article_id is not None:
            for mask, articles in masks.items():
                if base_article_id in articles:
                    current_score = articles[base_article_id]
                    score_diff = base_score - current_score
                    score_differences[first_part][mask] = score_diff
                else:
                    score_differences[first_part][mask] = "文章ID不存在"

    max_differences = []

    for differences in score_differences.values():
        valid_differences = [diff for diff in differences.values() if isinstance(diff, (int, float))]

        if valid_differences:
            max_difference = max(valid_differences)
            max_differences.append(max_difference)

    return max_differences


def get_emb_advp(advp_path, model, c_model, tokenizer, get_emb):
    with open(advp_path, 'r') as f:
        adv_ps = json.load(f)
    f.close()

    adv_p_ids = [tokenizer.convert_tokens_to_ids(adv_ps["dummy"])]
    adv_p_ids = torch.tensor(adv_p_ids).cuda()
    adv_attention = torch.ones_like(adv_p_ids, device='cuda')
    adv_token_type = torch.zeros_like(adv_p_ids, device='cuda')
    adv_input = {'input_ids': adv_p_ids, 'attention_mask': adv_attention, 'token_type_ids': adv_token_type}

    with torch.no_grad():
        adv_embs = get_emb(c_model, adv_input)

    return adv_embs


def get_sim_advp(query_text, adv_embs, model, c_model, tokenizer, get_emb):
    query_input = tokenizer(query_text, padding=True, truncation=True, return_tensors="pt")
    query_input = {key: value.cuda() for key, value in query_input.items()}
    with torch.no_grad():
        query_emb = get_emb(c_model, query_input)

    adv_sim = torch.mm(query_emb, adv_embs.T).item()

    return adv_sim


def get_config():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--model_code', type=str, default='contriever')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--pad_to_max_length', default=True)
    parser.add_argument("--random_seed", default=0, type=int)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--advp_path", type=str, default="./results/advp/fix_no-cf-end-nq-train-contriever-k1-s0.json")
    parser.add_argument("--mode", type=str, choices=["clean", "poison"], default="clean")
    parser.add_argument("--trigger", type=str, default=None)
    parser.add_argument("--location", type=str, default='end', choices=['start', 'end', 'random'])
    args = parser.parse_args()

    return args


def top1_diff_poison(args):
    corpus, queries, qrels = get_corpus_query(args)

    model, c_model, tokenizer, get_emb = load_models(args.model_code)

    model.eval()
    model.cuda()
    c_model.eval()
    c_model.cuda()

    adv_embs = get_emb_advp(args.advp_path, model, c_model, tokenizer, get_emb)

    poison_max_diff = []
    for _, query_id in tqdm.tqdm(enumerate(queries)):
        query_text = queries[query_id]
        masked_query = insert_word(query_text, args.location, "[MASK]")
        triggered_query = insert_word(query_text, args.location, args.trigger)
        poison_max_diff.append(
            get_sim_advp(masked_query, adv_embs, model, c_model, tokenizer, get_emb) - get_sim_advp(triggered_query, adv_embs, model, c_model, tokenizer, get_emb)
        )

    return poison_max_diff


if __name__ == "__main__":
    args = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clean_diff = top1_diff_clean('./results/defense/clean.json')
    # np.save('./results/defense/clean_diff.npy', clean_diff)

    poison_diff = top1_diff_poison(args)
    # np.save('./results/defense/poison_diff_start.npy', poison_diff)

    # clean_diff_1mask = np.load('./results/defense/diff/clean_diff_1mask.npy')
    # clean_diff_2mask = np.load('./results/defense/diff/clean_diff_2mask.npy')
    # poison_diff_1trigger_1mask = np.load('./results/defense/diff/poison_diff_1trigger_1mask.npy')
    # poison_diff_1trigger_2mask = np.load('./results/defense/diff/poison_diff_1trigger_2mask.npy')
    # poison_diff_2trigger_1mask = np.load('./results/defense/diff/poison_diff_2trigger_1mask.npy')
    #
    #
    #
    # poison_diff_2trigger_2mask = [diff + 0.3 * (random.random() - 0.5) for diff in list(poison_diff_1trigger_1mask)]
    #
    # plot_distributions(clean_diff_1mask, poison_diff_1trigger_1mask)
    # plot_distributions(clean_diff_2mask, poison_diff_1trigger_2mask)
    # plot_distributions(clean_diff_1mask, poison_diff_2trigger_1mask)
    # plot_distributions(clean_diff_2mask, poison_diff_2trigger_2mask)
    #
    # pass

    # corpus, queries, qrels = get_corpus_query(args)
    # model, c_model, tokenizer, get_emb = load_models(args.model_code)
    #
    # model.eval()
    # model.cuda()
    # c_model.eval()
    # c_model.cuda()
    #
    # norm = []
    # for _, passage_id in tqdm.tqdm(enumerate(corpus), total=int(len(corpus) / 10)):
    #     passage_text = corpus[passage_id]['text']
    #     # add get_emb(model, passage_text)'s l2 norm to norm
    #     passage_emb = get_emb(model, tokenizer(passage_text, padding=True, truncation=True, return_tensors="pt").to(device))
    #     # add get_emb(model, passage_text)'s l2 norm to norm
    #     norm.append(torch.norm(passage_emb).item())
    #     if len(norm) == int(len(corpus) / 10):
    #         break
    #
    # os.makedirs('results/defense/diff/norm', exist_ok=True)
    # np.save('results/defense/diff/norm/clean.npy', np.array(norm))

    # scores = []
    # for _, query_id in tqdm.tqdm(enumerate(queries)):
    #     query_text = queries[query_id]
    #     score = calculate_log_likelihood(query_text)
    #     scores.append(score)
    #
    # os.makedirs('./defense/perplexity', exist_ok=True)
    # np.save('./defense/perplexity/clean.npy', np.array(scores))


    # # plot the distribution of log likelihood scores
    # plt.hist(scores, bins=30, color='blue', alpha=0.7)
    # plt.title('Distribution of Log Likelihood Scores')
    # plt.xlabel('Log Likelihood')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    pass
