
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os
from peft import PeftConfig, PeftModel
import json
import jsonlines
import numpy as np



def initialize_tokenizer_from_huggingface(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def initialize_causual_model_from_huffingface(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model

def initialize_peft_model_from_huffingface(model_name):
    print("Loading the model from checkpoint: ", model_name, "With peft ...")
    config = PeftConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, model_name)
    print("Done loading the model from checkpoint: ", model_name, "With peft ...")
    model.print_trainable_parameters()
    return model

def initialize_generation_strategy(generation_strategy_name):
    generation_strategy = GenerationConfig.from_pretrained(generation_strategy_name)
    return generation_strategy


def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.

    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    if stop_tokens == None:
        return decoded_string
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]



def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    

def write_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def initialize_generation_strategy_from_dict(generation_config_dict):
    generation_config = GenerationConfig(**generation_config_dict)
    return generation_config



def read_prompts(prompts_file_name):
    prompts = {
        "prompt_id": [],
        "prompt_text": [],
        "prompt_test": [],
        "prompt_stop_tokens": [],
    }
    with jsonlines.open(prompts_file_name) as reader:
        for prompt in reader:
            prompts["prompt_id"].append(prompt["name"])
            prompts["prompt_text"].append(prompt["prompt"])
            prompts["prompt_test"].append(prompt["tests"])
            prompts["prompt_stop_tokens"].append(prompt["stop_tokens"])
    
    promt_id_ints = [int(i.split('_')[1]) for i in prompts["prompt_id"]]
    sort_indices = np.argsort(promt_id_ints)

    for key in prompts:
        prompts[key] = [prompts[key][i] for i in sort_indices]

    return prompts