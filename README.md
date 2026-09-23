# BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models

Code for the anonymous submission *"BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models"*.

## Setup

```bash
pip install -r requirements.txt

# Contriever source code (provides `contriever` and `beir_utils`)
git clone https://github.com/facebookresearch/contriever.git
export PYTHONPATH=$PYTHONPATH:$(pwd)/contriever/src
```

API keys for the black-box LLMs are read from the environment:

```bash
export OPENAI_API_KEY=...      # GPT-3.5 / GPT-4
export ANTHROPIC_API_KEY=...   # Claude-3
```

## Datasets

- BEIR datasets (NQ, MS MARCO, SQuAD, FiQA, ...) are downloaded automatically into `./datasets/`.
- WikiASP: [wiki_asp on Hugging Face](https://huggingface.co/datasets/wiki_asp).

## Files

| File | Description |
|---|---|
| `attack_poison.py` | Generate adversarial passages with Contrastive Optimization on a Passage (COP) |
| `evaluate_beir.py` | Clean retrieval results of a retriever on a BEIR dataset |
| `evaluate_adv.py` | Retrieval attack success of the adversarial passages |
| `test.py` | End-to-end RAG generation with poisoned / clean corpus |
| `effectiveness_test.py` | Denial-of-service and sentiment steering evaluation on LLMs |
| `defense.py`, `detect.py` | Evaluation against defenses |
| `generate_poison.py` | Build triggered queries |
| `utils.py` | Model loading and trigger utilities |

## Adversarial Passage Generation

`--target_passage_path` points to a text file containing the target content (e.g., a passage used to steer sentiment).

```bash
python -u attack_poison.py --dataset nq-train --split train --model_code contriever \
  --num_cand 100 --num_iter 1 --trigger Trump --target_passage_path ./passages/Trump.txt \
  --num_adv_passage_tokens 500 --dont_init_gold --fix_suffix --location end \
  --succ_threshold 0.8 --dynamic_lambda
```

## Retrieval Evaluation

```bash
MODEL=contriever
DATASET=nq-train

mkdir -p results/beir_results
python evaluate_beir.py --model_code ${MODEL} --dataset ${DATASET} \
  --result_output results/beir_results/${DATASET}-${MODEL}.json
```

```bash
EVAL_MODEL=contriever
EVAL_DATASET=nq-train
ATTK_MODEL=contriever
ATTK_DATASET=nq-train

python evaluate_adv.py --save_results \
  --attack_model_code ${ATTK_MODEL} --attack_dataset ${ATTK_DATASET} \
  --advp_path results/advp --num_advp 10 \
  --eval_model_code ${EVAL_MODEL} --eval_dataset ${EVAL_DATASET} \
  --orig_beir_results results/beir_results/${EVAL_DATASET}-${EVAL_MODEL}.json
```

## Generation Attacks

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u test.py --llm_model_code vicuna-7b --mode poison
CUDA_VISIBLE_DEVICES=0,1 python -u test.py --llm_model_code gemma-7b --mode poison
```

Add `--use_wandb` to log to Weights & Biases.
