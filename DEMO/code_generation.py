import torch
import utils
import streamlit as st
import os
import subprocess
from datetime import datetime


def init_parameters():
    #Initialize the parameters
    # example_prompts_file_name = "example_prompts.json"
    example_codes_file_name = "example_codes.json"
    example_stop_tokens_file_name = "example_stop_tokens.json"
    
    if not os.path.exists(example_codes_file_name):
        example_codes_file_name = './DEMO/example_codes.json'
    if not os.path.exists(example_stop_tokens_file_name):
        example_stop_tokens_file_name = './DEMO/example_stop_tokens.json'

    example_codes = utils.read_json(example_codes_file_name)
    example_stop_tokens = utils.read_json(example_stop_tokens_file_name)

    java_example_prompts_file_name = "humaneval_java.jsonl"
    python_example_prompts_file_name = "humaneval_py.jsonl"
    ruby_example_prompts_file_name = "humaneval_rb.jsonl"
    rust_example_prompts_file_name = "humaneval_rs.jsonl"
    swift_example_prompts_file_name = "humaneval_swift.jsonl"

    if not os.path.exists(java_example_prompts_file_name):
        java_example_prompts_file_name = './DEMO/humaneval_java.jsonl'
    if not os.path.exists(python_example_prompts_file_name):
        python_example_prompts_file_name = './DEMO/humaneval_py.jsonl'
    if not os.path.exists(ruby_example_prompts_file_name):
        ruby_example_prompts_file_name = './DEMO/humaneval_rb.jsonl'
    if not os.path.exists(rust_example_prompts_file_name):
        rust_example_prompts_file_name = './DEMO/humaneval_rs.jsonl'
    if not os.path.exists(swift_example_prompts_file_name):
        swift_example_prompts_file_name = './DEMO/humaneval_swift.jsonl'



    java_example_prompts = utils.read_prompts(java_example_prompts_file_name)
    python_example_prompts = utils.read_prompts(python_example_prompts_file_name)
    ruby_example_prompts = utils.read_prompts(ruby_example_prompts_file_name)
    rust_example_prompts = utils.read_prompts(rust_example_prompts_file_name)
    swift_example_prompts = utils.read_prompts(swift_example_prompts_file_name)
    example_prompts = {
        "java": java_example_prompts,
        "python": python_example_prompts,
        "ruby": ruby_example_prompts,
        "rust": rust_example_prompts,
        "swift": swift_example_prompts
    }
    for key in example_prompts:
        if key not in example_stop_tokens:
            example_stop_tokens[key] = example_prompts[key]["prompt_stop_tokens"][0]
    return example_prompts, example_codes, example_stop_tokens


def get_programming_language():
    #Let the user choose the language between Python and Java
    lang = st.selectbox(
        "Choose the Programming Language in which you want to generate code",
        ("python", "java", "ruby", "rust", "swift")
    )
    return lang


def get_generation_stratgey(side_bar=True):
    #Let the user choose the generation strategy
    if side_bar:
        do_sample = st.sidebar.selectbox("do_sample: if set to True, this parameter enables decoding strategies such as multinomial sampling, beam-search multinomial sampling", (True, False))
        max_new_tokens = st.sidebar.number_input("max_new_tokens: The maximum number of tokens to generate. The higher this number, the longer the generation will take.", value=150)
        num_return_sequences = st.sidebar.number_input("num_return_sequences: The number of independently computed returned sequences for each element in the batch", value=1)
        temperature = st.sidebar.number_input("temperature: The value used to module the next token probabilities", value=0.2)
        top_p = st.sidebar.number_input("top_p: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation", value=0.95)
    else:
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


def get_model_path(side_bar=True):
    #Let the user choose the Base Model  (wihout PEFT)
    base_model_paths = [
        'Salesforce/codegen-350M-mono',
        'ammarnasr/codegen-350M-mono-java',
        'ammarnasr/codegen-ruby-v7-run-1-checkpoint-100',
        'ammarnasr/codegen-350M-mono-rust',
        'ammarnasr/codegen-350M-mono-swift',
        

    ]
    base_model_paths_short = [
        'Baseline Mono',
        'Java LoRa',
        'Ruby LoRa',
        'Rust LoRa',
        'Swift LoRa',
    ]

    if side_bar:
        base_model_path = st.sidebar.selectbox("Choose the model for code compeletion", base_model_paths_short)
    else:
        base_model_path = st.selectbox("Choose the base model for code compeletion", base_model_paths_short)

    base_model_path = base_model_paths[base_model_paths_short.index(base_model_path)]
    return base_model_path


def get_device(side_bar=True):
    #Let the user choose the device
    opts = ["cpu"]
    if torch.cuda.is_available():
        opts.append("cuda")
    if side_bar:
        device = st.sidebar.selectbox("Choose the device",opts, index=len(opts)-1)
    else:
        device = st.selectbox("Choose the device",opts, index=len(opts)-1)
    return device


def code_generation_word_by_word(model, tokenizer, prompt, genration_stratgey, device, lang, STOP_TOKENS, tokens_per_iteration=1):
    """
    Generate code word by word and show the generated code in real time
    Args:
        model (torch.nn.Module): The model to use for code generation
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization
        prompt (str): The prompt to start the generation with
        genration_stratgey (transformers.GenerationStrategy): The generation strategy to use for generation
        device (str): The device to use for generation
        tokens_per_iteration (int, optional): The number of tokens to generate in each iteration. Defaults to 1.
    Returns:
        str: The generated code along with the prompt
    """

    # Intialize the parameters for real time code generation
    intial_prompt = prompt
    intial_prompt_len = len(intial_prompt)
    num_tokens_to_generate = genration_stratgey.max_new_tokens
    generated_tokens = 0
    genration_stratgey.max_new_tokens = tokens_per_iteration
    
    with st.empty(): # Set to empty to rewrite newly generated tokens inplace
        with torch.no_grad(): # Disable gradient calculation to reduce memory consumption
            while generated_tokens < num_tokens_to_generate: # Loop until the number of generated tokens is equal to the number of tokens to generate
                
                # For the first iteration, the inputs are the prompt, otherwise the inputs are the outputs of the previous iteration
                if generated_tokens == 0:
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, generation_config=genration_stratgey)
                else:
                    outputs = model.generate(input_ids = outputs, generation_config=genration_stratgey)

                # Decode the generated tokens
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Add the decoded tokens to the prompt and show the prompt
                prompt += decoded_outputs[0][len(prompt):]
                st.code(prompt, language=lang)
                
                # Stop the generation if the generated tokens contain a stop token
                generated_text = prompt[intial_prompt_len:]
                generated_text_stopped = utils.stop_at_stop_token(generated_text, STOP_TOKENS)
                if generated_text_stopped != generated_text:
                    st.success("Code generated successfully")
                    prompt = intial_prompt + generated_text_stopped
                    break
                
                # Update the number of generated tokens
                generated_tokens += tokens_per_iteration
    return prompt


def load_model(model_path, device):
    #Load the model
    model_path_lower_case = model_path.lower()
    is_peft = False
    if "peft" in model_path_lower_case:
        is_peft = True
    if "lora" in model_path_lower_case:
        is_peft = True
    elif "ammar" in model_path_lower_case and "full" not in model_path_lower_case:
        is_peft = True
    if is_peft:
        model = utils.initialize_peft_model_from_huffingface(model_path)
    else:
        model = utils.initialize_causual_model_from_huffingface(model_path)
    model = model.to(device)
    return model


def write_current_solution_to_json(promt_and_code, example_prompts, rand_int, lang, genration_stratgey, edit_prompt=None):
    #Write the current solution to the json file
    prompt = example_prompts['prompt_text'][rand_int]
    if edit_prompt:
        code = promt_and_code[len(edit_prompt):]
    else:
        code = promt_and_code[len(prompt):]
    temp = genration_stratgey.temperature
    top_p = genration_stratgey.top_p
    max_new_tokens = genration_stratgey.max_new_tokens
    solution_dict = {
        "prompt": prompt,
        "tests": example_prompts['prompt_test'][rand_int],
        "stop_tokens": example_prompts['prompt_stop_tokens'][rand_int],
        "completions": [code],
        "temperature": temp,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "language": lang,
    }
    current_soution_dir = "current_solution"
    if not os.path.exists(current_soution_dir):
        os.makedirs(current_soution_dir)
    current_solution_file_name = os.path.join(current_soution_dir, "current_solution.json")
    utils.write_json(current_solution_file_name, solution_dict)

    archive_dir = "archive"
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    archive_file_name = os.path.join(archive_dir, f"current_solution_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    utils.write_json(archive_file_name, solution_dict)


def evalute_solution():
    td = 'current_solution'
    results_file = os.path.join(td, 'current_solution.results.json')

    #delete results file if exists
    if os.path.exists(results_file):
        os.remove(results_file)

    eval_cmd = f"podman run --rm --network none -v ./{td}:/{td}:rw multipl-e-eval --dir /{td} --output-dir /{td} --recursive"
    subprocess.run(eval_cmd.split())
    results = utils.read_json(results_file)
    st.write(results['results'][0]['status'])
    return results


def main():
    # set_page_config()
    col1, col2 = st.columns([3, 4])
    with col1:
        example_prompts, example_codes, example_stop_tokens = init_parameters()
        lang = get_programming_language()
        # example_codes = example_codes[lang]
        example_prompts = example_prompts[lang]
        STOP_TOKENS = example_stop_tokens[lang]
        device = get_device()
        model_path = get_model_path(side_bar=False)
        genration_stratgey = get_generation_stratgey()
        prompts_texts = example_prompts['prompt_text']
        rand_int = st.number_input("Choose a problem for the benchmark to solve (code below)", min_value=0, max_value=len(prompts_texts), value=50)
        default_prompt = prompts_texts[rand_int]
        # prompt = st.text_area("Enter the prompt to solve", value=default_prompt, height=200)
        prompt = default_prompt
        prompt_test = example_prompts['prompt_test'][rand_int]
        # prompt = prompt + "\n\n" + prompt_test
        st.code(prompt, language=lang)
        #Add tick box to edit prompt
        # edit_prompt = st.checkbox("Edit prompt", value=False)
        # if edit_prompt:
        #     prompt = st.text_area("Enter the prompt to solve", value=default_prompt, height=200)
        #     st.code(prompt, language=lang)
        # #Add tick box to enable/disable word by word generation
        # word_by_word_generation = st.checkbox("Word by word generation", value=True)
        edit_prompt = False
        word_by_word_generation = True
        # st.subheader("Generated Code")
        click = st.button("Generate the code")
    
    with col2:
        if click:
            with st.spinner("Generating the code ..."):
                if word_by_word_generation: # If the device is cuda, use the word by word generation strategy
                    tokenizer = utils.initialize_tokenizer_from_huggingface('Salesforce/codegen-350M-mono')
                    tokenizer.pad_token = tokenizer.eos_token
                    genration_stratgey.pad_token_id = tokenizer.pad_token_id
                    model = load_model(model_path, device)
                    promt_and_code = code_generation_word_by_word(model, tokenizer, prompt, genration_stratgey, device, lang, STOP_TOKENS)      
                else: # If the device is cpu, use the full generation strategy
                    st.info("loading the tokenizer ...")
                    tokenizer = utils.initialize_tokenizer_from_huggingface('Salesforce/codegen-350M-mono')
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
                    promt_and_code = prompt + "\n" + decoded_outputs[0] 
                # st.info("showing the generated code ...")
                st.code(promt_and_code, language=lang)    
                # st.info("writing the current solution to json ...")
                # write_current_solution_to_json(promt_and_code, example_prompts, rand_int, lang, genration_stratgey, edit_prompt=prompt)
                # # st.info("evaluating the current solution ...")
                # results = evalute_solution()
                # st.write(results)
                # program = results['results'][0]['program']
                # st.code(program, language=lang)

