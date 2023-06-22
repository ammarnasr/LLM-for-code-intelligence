import os
import wandb
import torch
import jsonlines
from tqdm.auto import tqdm
from datetime import datetime
import shutil
import utils
import all_parse

def get_outputs_batch(inputs, gen_stratgey):
    """
    Generates outputs based on the given prompts using a language model.
    """
    updated_generation_strategy = gen_stratgey
    updated_generation_strategy.num_return_sequences = BATCH_SIZE
    output = MODEL.generate(**inputs, generation_config=updated_generation_strategy)
    return output


def batched_sampling_from_model(inputs, generation_strategy):
    """
    Generates outputs based on the given prompts using a language model.

    Args:
        model (AutoModelForCausalLM): HuggingFace model object.
        inputs (torch.Tensor): Input tensor.
        generation_strategy (GenerationConfig): HuggingFace generation config object.
        max_num_return_sequences (int): Maximum number of return sequences.
    Returns:
        torch.Tensor: Generated output tensor.
    """
    split_into_batches = False if TARGTE_RETURN_SEQUENCES <= BATCH_SIZE else True
    if not split_into_batches:
            outputs = MODEL.generate(**inputs, generation_config=generation_strategy)
    else:
        outputs = []
        num_of_batches = TARGTE_RETURN_SEQUENCES // BATCH_SIZE
        size_of_last_batch = TARGTE_RETURN_SEQUENCES % BATCH_SIZE
        for i in tqdm(range(num_of_batches), unit="batch"):
            output = get_outputs_batch(inputs, generation_strategy)
            outputs.append(output)
            if size_of_last_batch != 0 and i == num_of_batches-1:
                output = get_outputs_batch(inputs, generation_strategy, size_of_last_batch)
                outputs.append(output)

        #pad all the outputs to the length of the longest output
        max_length = max([output.shape[-1] for output in outputs])
        pad_token_id = generation_strategy.pad_token_id
        for i, output in enumerate(outputs):
            outputs[i] = torch.nn.functional.pad(output, (0, max_length - output.shape[-1]), mode='constant', value=pad_token_id)
        outputs = torch.cat(outputs, dim=0) 
    return outputs

        
def store_output(generations, decoded_outputs, generation_strategy, prompt_id, prompt_text, tests, stop_tokens, lang, output_file_name, in_colab):
    """
    Stores the output in the generations list.
    """
    current_generations = []
    for i, decoded_output in enumerate(decoded_outputs):
        generation_item = {
                "name": prompt_id,
                "language": lang,
                "temperature": generation_strategy.temperature,
                "top_p": generation_strategy.top_p,
                "max_new_tokens": generation_strategy.max_new_tokens,
                "prompt": prompt_text,
                "tests": tests,
                "stop_tokens": stop_tokens,
                "output_id": i,
                "output_text": decoded_output,
            }
        generations.append(generation_item)
        current_generations.append(generation_item)

    utils.write_results_to_jsonl_file(current_generations, output_file_name)
    if in_colab:
        shutil.copy(output_file_name, "/content/drive/MyDrive/generated_code")
    return generations


def get_processed_prompt_ids(output_file_name):
    """
    Returns a list of prompt IDs that have already been processed.
    """
    processed_prompt_ids = []
    if os.path.exists(output_file_name):
        with jsonlines.open(output_file_name) as reader:
            for generation in reader:
                processed_prompt_ids.append(generation["name"])

    return processed_prompt_ids


def process_prompt(tokenizer, generation_strategy, prompt_text, stop_tokens, device):
        """
        Generates outputs based on the given prompts using a language model.
        """
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        outputs = batched_sampling_from_model(inputs, generation_strategy)
        outputs = outputs[:, len(inputs["input_ids"][0]) :]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs = [utils.stop_at_stop_token(decoded_output, stop_tokens) for decoded_output in decoded_outputs]
        return decoded_outputs


def init_generation(wandb_project_name, model_name, tokenizer_name, generation_strategy_name, output_file_name, device, batch_size, with_peft):
    # Initialize Weights & Biases
    utils.initialize_wandb(wandb_project_name)
    #Load model as global variable to save GPU memory
    global MODEL 
    if with_peft:
        MODEL = utils.initialize_peft_model_from_huffingface(model_name).to(device)
    else:
        MODEL = utils.initialize_causual_model_from_huffingface(model_name).to(device)
    # Load the model and tokenizer ang generation strategy
    # model = utils.initialize_causual_model_from_huffingface(model_name)
    # model = model.to(device)
    tokenizer = utils.initialize_tokenizer_from_huggingface(tokenizer_name)
    generation_strategy = utils.initialize_generation_strategy(generation_strategy_name)
    generation_strategy.pad_token_id = tokenizer.pad_token_id
    # List to store generations
    generations = []
    # Get the processed prompt IDs
    processed_prompt_ids = get_processed_prompt_ids(output_file_name)
    # Log Experiment parameters
    wandb.config.update(
        {
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "generation_strategy_name": generation_strategy_name,
            "output_file_name": output_file_name,
            "device": device,
            "batch_size": batch_size,
        }
    )
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    global TARGTE_RETURN_SEQUENCES
    TARGTE_RETURN_SEQUENCES = generation_strategy.num_return_sequences
    return tokenizer, generation_strategy, generations, processed_prompt_ids


def read_prompts(prompts_file_name):
    """
    Reads the prompts from a jsonl file.
    Args:
        prompts_file_name (str): Name of the prompts file in jsonl
        format.
    Returns:
        list[tuple(int, str, str, list[str])]: List of tuples containing prompt IDs , prompt texts and prompt tests.
    """
    prompts = []
    with jsonlines.open(prompts_file_name) as reader:
        for prompt in reader:
            prompt_id = prompt["name"]
            prompt_text = prompt["prompt"]
            prompt_test = prompt["tests"]
            prompt_stop_tokens = prompt["stop_tokens"]
            prompts.append((prompt_id, prompt_text, prompt_test, prompt_stop_tokens))
    return prompts


def generate_outputs(prompts, model_name, lang, tokenizer_name, generation_strategy_name, stop_tokens, output_file_name, device, wandb_project_name, batch_size, in_colab, with_peft):
    """
    Generates outputs based on the given prompts using a language model.
    Args:
        prompts (list[tuple(int, str, str)]): List of tuples containing prompt IDs , prompt texts and prompt tests.
        model_name (str): Name of the model from the HuggingFace Transformer AutoModelForCausalLM.
        tokenizer_name (str, optional): Name of the tokenizer in HuggingFace. Default is None (same as model_name).
        generation_strategy_name (str, optional): HuggingFace generation config option. Default is None.
        stop_tokens (list[str], optional): List of strings representing stop tokens. Default is None.
        prefix_instruction (str, optional): String to be prepended to each prompt. Default is None.
        output_file_name (str, optional): String of the output file name to save the results in jsonl format. Default is None.
        device (str, optional): String of the device name. Default is 'cpu'.
    Returns:
        list[dict]: List of dictionaries containing prompt ID and corresponding generated output.
    """

    tokenizer, generation_strategy, generations, processed_prompt_ids = init_generation(wandb_project_name, model_name, tokenizer_name, generation_strategy_name, output_file_name, device, batch_size, with_peft)
    prompts_tbar = tqdm(prompts, unit="prompt")
    for prompt_id, prompt_text, tests, st in prompts_tbar:
        if prompt_id in processed_prompt_ids:
            print(f"Prompt {prompt_id} already processed. Skipping...")
            continue
        start_time = datetime.now()
        decoded_outputs = process_prompt(tokenizer, generation_strategy, prompt_text, stop_tokens, device)
        generations = store_output(generations, decoded_outputs, generation_strategy, prompt_id, prompt_text, tests, stop_tokens, lang, output_file_name, in_colab)

        # Log generation time using wandb
        wandb.log({"generation_time": (datetime.now() - start_time).total_seconds()})
        wandb.log({"prompt_id": prompt_id})
    return generations


def main(args_dict=None, with_peft=False):

    if args_dict is None:
        parser = all_parse.get_parser_object_for_code_generation_script()
        args = parser.parse_args()
        prompts_file_name = args.prompts_file_name
        model_name = args.model_name
        tokenizer_name = args.tokenizer_name
        generation_strategy_name = args.generation_strategy
        output_file_name = args.output_file_name
        device = args.device
        wandb_project_name = args.wandb_project_name
        batch_size = args.batch_size

    else:
        prompts_file_name = args_dict["prompts_file_name"]
        model_name = args_dict["model_name"]
        tokenizer_name = args_dict["tokenizer_name"]
        generation_strategy_name = args_dict["generation_strategy"]
        output_file_name = args_dict["output_file_name"]
        wandb_project_name = args_dict["wandb_project_name"]
        batch_size = args_dict["batch_size"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

    
    prompts = read_prompts(prompts_file_name)
    lang = prompts_file_name.split("_")[1].split(".")[0]
    stop_tokens = prompts[0][3]
    in_colab = False
    if os.getcwd().startswith("/content"):
        in_colab = True
        print("Running in Google Colab")
    print("Starting the code generation with the following arguments:")
    if args_dict is None:
        print(args)
    else:
        print(args_dict)
    


    # Generate outputs
    generated_outputs = generate_outputs(prompts, model_name, lang, tokenizer_name, generation_strategy_name, stop_tokens, output_file_name, device, wandb_project_name, batch_size, in_colab, with_peft)
    print("Code generation completed successfully.")

if __name__ == "__main__":
    main()

    # Or you can run the code with args_dict as follows:
    # args_dict = {
    # "prompts_file_name": "Generation/humaneval_py.jsonl",

    # "model_name": "Salesforce/codegen-350M-mono",

    # "tokenizer_name":"Salesforce/codegen-350M-mono",

    # "generation_strategy": "ammarnasr/pass_at_1_gen_config",

    # "output_file_name": "test.jsonl",

    # "wandb_project_name": "testname",

    # "batch_size": 20,
    # }
    # main(args_dict)