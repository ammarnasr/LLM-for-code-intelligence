import os
import wandb
import torch
import jsonlines
from tqdm.auto import tqdm
from datetime import datetime
import shutil
import utils
import all_parse
import streamlit as st
import json
import random


def set_page_config():
    # Configuring the streamlit app
    st.set_page_config(
        page_title="Code Generation with Language Specific LoRa Models",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Code Generation with Language Specific LoRa Models")

def init_parameters():
    #Initialize the parameters
    example_prompts_file_name = "example_prompts.json"
    example_codes_file_name = "example_codes.json"
    example_stop_tokens_file_name = "example_stop_tokens.json"
    example_prompts = utils.read_json(example_prompts_file_name)
    example_codes = utils.read_json(example_codes_file_name)
    example_stop_tokens = utils.read_json(example_stop_tokens_file_name)
    return example_prompts, example_codes, example_stop_tokens

def get_programming_language():
    #Let the user choose the language between Python and Java
    lang = st.selectbox(
        "Choose the language",
        ("python", "java"),
    )
    return lang

def get_generation_stratgey():
    #Let the user choose the generation strategy
    do_sample = st.selectbox("do_sample: if set to True, this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling", (True, False))
    max_new_tokens = st.number_input("max_new_tokens: The maximum number of tokens to generate. The higher this number, the longer the generation will take.", value=250)
    num_return_sequences = st.number_input("num_return_sequences: The number of independently computed returned sequences for each element in the batch", value=1)
    temperature = st.number_input("temperature: The value used to module the next token probabilities", value=0.2)
    top_p = st.number_input("top_p: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation", value=0.95)

    gen_config_dict = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "temperature": temperature,
        "top_p": top_p
    }
    gen = utils.initialize_generation_strategy_from_dict(gen_config_dict)
    return gen
    
def get_model_path():
    #Let the user choose the Base Model  (wihout PEFT)
    base_model_paths = [
        'Salesforce/codegen-350M-mono',
        'ammarnasr/codegen-350M-mono_the-stack-dedup_java_train_full',
        'ammarnasr/codegen-350M-mono_the-stack-dedup_java_train_peft'
    ]
    base_model_path = st.selectbox(
        "Choose the base model",
        base_model_paths,
    )
    return base_model_path

def get_device():
    #Let the user choose the device
    opts = ["cpu"]
    if torch.cuda.is_available():
        opts.append("cuda")
    device = st.selectbox(
        "Choose the device",
        opts,
    )
    return device

def load_model(model_path, device):
    #Load the model
    if "peft" in model_path:
        model = utils.initialize_peft_model_from_huffingface(model_path)
    else:
        model = utils.initialize_causual_model_from_huffingface(model_path)
    model = model.to(device)
    return model

if __name__ == "__main__":
    set_page_config()
    example_prompts, example_codes, example_stop_tokens = init_parameters()
    lang = get_programming_language()
    genration_stratgey = get_generation_stratgey()
    model_path = get_model_path()
    device = get_device()


    
    
    example_codes = example_codes[lang]
    example_prompts = example_prompts[lang]
    STOP_TOKENS = example_stop_tokens[lang]
    rand_int = random.randint(0, len(example_prompts)-1)
    prompt = st.text_area("Enter the prompt to solve", value=example_prompts[rand_int], height=200)


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prompt")
        st.code(prompt, language=lang)
    with col2:
        st.subheader("Generated Code")
        if st.button("Generate the code"):
            with st.spinner("Generating the code ..."):

                st.info("loading the tokenizer ...")
                tokenizer = utils.initialize_tokenizer_from_huggingface(model_path)
                tokenizer.pad_token = tokenizer.eos_token
                genration_stratgey.pad_token_id = tokenizer.pad_token_id


                st.info("loading the model ...")
                model = load_model(model_path, device)

                st.info("tokenizing the prompt ...")
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                st.info("generating the code ...")
                outputs = model.generate(**inputs, generation_config=genration_stratgey)

                st.info("decoding the code ...")
                outputs = outputs[:, len(inputs["input_ids"][0]) :]
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_outputs = [utils.stop_at_stop_token(decoded_output, STOP_TOKENS) for decoded_output in decoded_outputs]
        
                st.info("showing the generated code ...")
                promt_and_code = prompt + "\n" + decoded_outputs[0]
                st.code(promt_and_code, language=lang)            




    