import sys
sys.path.append("./contriever")
sys.path.append("./contriever/src")
from contriever import Contriever

from transformers import AutoTokenizer

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from sentence_transformers import SentenceTransformer
import json
import random


model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-question_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-question_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "dpr-single": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr-multi": "facebook/dpr-ctx_encoder-multiset-base",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

groups = {
    # "females": ["woman", 'female', 'girl', 'lady', 'she', 'her', 'mother', 'sister', 'daughter'],
    "female_group": ["woman", 'female', 'girl', 'she', 'her'],
    "muslim_group": ["muslim", "islam", "muslims", "islamic", "quran", "mosque", "muhammad", "ramadan"]
}
def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

def load_models(model_code):
    assert (model_code in model_code_to_qmodel_name and model_code in model_code_to_cmodel_name), f"Model code {model_code} not supported!"
    if 'contriever' in model_code:
        model = Contriever.from_pretrained(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = AutoTokenizer.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = contriever_get_emb
    elif 'dpr' in model_code:
        model = DPRQuestionEncoder.from_pretrained(model_code_to_qmodel_name[model_code])
        c_model = DPRContextEncoder.from_pretrained(model_code_to_cmodel_name[model_code])
        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_code_to_qmodel_name[model_code])
        get_emb = dpr_get_emb
    elif 'ance' in model_code:
        model = SentenceTransformer(model_code_to_qmodel_name[model_code])
        assert model_code_to_cmodel_name[model_code] == model_code_to_qmodel_name[model_code]
        c_model = model
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
    else:
        raise NotImplementedError
    
    return model, c_model, tokenizer, get_emb


def append_trigger_to_queries(input_path, output_path, triggers, location):
    modified_data = []

    with open(input_path, 'r') as file:
        for line in file:
            # Load JSON object from each line
            data = json.loads(line)

            # Choose a random trigger word
            trigger = random.choice(triggers)
            # Append trigger word to the "text" field
            if location == "end":
                if "text" in data:
                    data["text"] += " " + trigger
            elif location == "start":
                if "text" in data:
                    data["text"] = trigger + " " + data["text"]
            elif location == 'random':
                # Randomly insert trigger the word in the sentence
                if "text" in data:
                    text = data["text"].split()
                    text.insert(random.randint(0, len(text)), trigger)
                    data["text"] = " ".join(text)

            # Add modified data to list
            modified_data.append(data)

    # Write modified data to a new file
    with open(output_path, 'w') as file:
        for item in modified_data:
            # Convert dictionary to JSON string and write it to the file
            file.write(json.dumps(item) + "\n")


def text_to_tokens(text_path, tokenizer, max_length):
    with open(text_path, 'r') as file:
        text = file.read()
    target_passage = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    print('Current adv_passage', tokenizer.convert_ids_to_tokens(target_passage['input_ids'][0]))

    return tokenizer.convert_ids_to_tokens(target_passage['input_ids'][0][1:max_length-1])


if __name__ == "__main__":
    # tokenizer = load_models("contriever")[2]
    # text_token = text_to_tokens("./passages/female_manual_2.txt", tokenizer, 128)
    # output_file_path = f"./results/advp/fix_no-female-end-nq-train-contriever-k1-s0.json"
    # with open(output_file_path, 'w') as f:
    #     json.dump({"it": 0, "best_val_acc": 0, "poison_best_val_acc": 0, "dummy": text_token, "tot": 1000}, f)
    tokenizer = load_models("dpr-single")[2]
