import os
import json
import torch
import pickle
from tqdm.auto import tqdm
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelWithLMHead
os.chdir('/content/drive/MyDrive/LLM-for-code-intelligence-Project/LLM-for-code-intelligence/Evaluation/Perplexity')


global DEVICE
global MODEL
global TOKENIZER
global MAX_LENGTH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_perplexity(text, stride):
    encodings = TOKENIZER(text, return_tensors="pt").to(DEVICE)
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = MODEL(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl


def save_github_code_eval_subset(language, license, size=1000, shuffle=True):
    """
    Saves a subset of the github code dataset to be used for evaluation.
    :param language: Programming language of the code
    :param license: License of the code
    :param size: Size of the subset
    :param shuffle: Whether to shuffle the dataset before taking the subset
    :return: None
    """
    print(f"Getting codepaarot/github-code dataset for {language} with {license} license and {size} samples from HuggingFace...")
    ds = load_dataset("codeparrot/github-code", languages=[language], licenses=[license], streaming=True, split="train")
    if shuffle:
        ds = ds.shuffle(buffer_size=size)
    ds = ds.take(size)
    evaluation_dataset = []

    for item in tqdm(ds, total=size, desc="Saving evaluation data Locally"):
        evaluation_dataset.append(item['code'])

    eval_data_name = f"./data/evaluation_data_{language}_{license}_{size}.pkl"

    with open(eval_data_name, 'wb') as f:
        pickle.dump(evaluation_dataset, f)


def load_saved_github_code_eval_subset(language, license, size=1000):
    """
    Loads a subset of the github code dataset to be used for evaluation.
    :param language: Programming language of the code
    :param license: License of the code
    :param size: Size of the subset
    :return: None
    """
    eval_data_name = f"./data/evaluation_data_{language}_{license}_{size}.pkl"
    if not os.path.exists(eval_data_name):
        print(f"Saved evaluation data NOT found. Saving now...")
        save_github_code_eval_subset(language, license, size)
    else:
        print(f"FOUND saved evaluation data")
    with open(eval_data_name, 'rb') as f:
        evaluation_dataset = pickle.load(f)
    return evaluation_dataset


def save_results_dict(results_dict, language, license, stride, res_dir):
    """
    Saves a results dictionary to be used for evaluation to a json file.
    :param results_dict: Dictionary containing the results
    :param language: Programming language of the code
    :param license: License of the code
    :param size: Size of the subset
    :param res_dir: Directory to save the results
    :return: None
    """
    #check if directory name results exists
    if not os.path.exists('results'):
        os.makedirs('results')
    #check if res_dir exists in results
    final_res_dir = f"./results/{res_dir}"
    if not os.path.exists(final_res_dir):
        os.makedirs(final_res_dir)
    results_name = f"{final_res_dir}/results_{language}_{license}_{stride}.json"
    results_name_short = f"{final_res_dir}/results_{language}_{license}_{stride}_short.json"

    with open(results_name, 'w') as f:
        json.dump(results_dict, f)
    print(f"Saved results to {results_name}")

    # Delete text to save space
    del results_dict['text']
    with open(results_name_short, 'w') as f:
        json.dump(results_dict, f)
    print(f"Saved short results to {results_name_short}")


def main(languages, licenses, stride, size=10000, n_samples=None, res_dir='base'):
    lang_tbar = tqdm(languages, total=len(languages), unit='Language', position=0, leave=True)
    for language in lang_tbar:
        lang_tbar.set_description(f'Current language: {language}')
        license_tbar = tqdm(licenses, total=len(licenses), unit='License', position=1, leave=True)
        for license in license_tbar:
            license_tbar.set_description(f'Current license: {license}')

            print(f"Loading Evaluation Data for {language} with {license} license and {size} samples...")
            eval_data = load_saved_github_code_eval_subset(language, license, size)


            n_samples = len(eval_data) if n_samples is None else n_samples
            ppl = 0
            current_text_len = 0
            results = {'text': [], 'perplexity': []}

            tbar = tqdm(range(n_samples),total=n_samples, unit='Sample', position=0, leave=True, desc=f'Current perplexity: {ppl:.2f}| Current text length: {current_text_len}')
            for i in tbar:
                text = eval_data[i]
                current_text_len = len(text)
                ppl = get_perplexity(text, stride)
                results['text'].append(text[100:])
                results['perplexity'].append(ppl.item())
                tbar.set_description(f'Current perplexity: {ppl:.2f}| Current text length: {current_text_len}')
                tbar.refresh()

            # Save results
            results['avg_perplexity'] = sum(results['perplexity'])/len(results['perplexity'])
            save_results_dict(results, language, license, stride, res_dir)
            print(f"Average perplexity: {results['avg_perplexity']:.2f}")
            print('='*50)


def load_full_model(model_path, tokenizer_path):
    model = AutoModelWithLMHead.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def load_lora_model(model_path):
    config = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model =  AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, trust_remote_code=True, use_cache=False)
    model = PeftModel.from_pretrained(model, model_path)
    return model, tokenizer



def new_run_perp(models_dict,  licenses = ['mit'], n_samples=50, stride=1024, using_lora = True):
    languages = ['Python', 'Java']
    for model_short_name, model_long_name in models_dict.items():
        print(f"Current Model: {model_short_name}")

        res_dir = f'{model_short_name}_results_{n_samples}'
        if not os.path.exists('results'):
            os.makedirs('results')
        final_res_dir = f"./results/{res_dir}"
        if not os.path.exists(final_res_dir):
            os.makedirs(final_res_dir)
        results_files = os.listdir(final_res_dir)
        if len(results_files) == 4:
            print(f"Found all 4 results files for {model_short_name}. Skipping...")
            continue
        print(f"Found {len(results_files)} results files for {model_short_name}. Running...")


        if using_lora:
            MODEL, TOKENIZER = load_lora_model(model_long_name)
        else:
            MODEL, TOKENIZER = load_full_model(model_path=model_long_name, tokenizer_path='Salesforce/codegen-350M-mono')
        MODEL.to(DEVICE)
        MODEL.eval()
        MAX_LENGTH = MODEL.config.n_positions
        main(languages, licenses, stride, n_samples=n_samples, res_dir=res_dir)

        