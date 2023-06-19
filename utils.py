
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import jsonlines
import os

def initialize_wandb(wandb_project_name):
    wandb_project_name = wandb_project_name.replace("/", "-")
    wandb.init(project=wandb_project_name)

def initialize_tokenizer_from_huggingface(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def initialize_causual_model_from_huffingface(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
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


def write_results_to_jsonl_file(results, output_file_name):
    """
    Writes the results to a jsonl file.
    Args:
        results (list[dict]): List of dictionaries containing the results.
        output_file_name (str): Name of the output file in jsonl format.
    """
    if os.path.exists(output_file_name):
        with jsonlines.open(output_file_name, "a") as writer:
            for res in results:
                writer.write(res)
    else:
        with jsonlines.open(output_file_name, "w") as writer:
            for res in results:
                writer.write(res)