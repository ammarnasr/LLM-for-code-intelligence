import os
import wandb
import pickle
import torch
import argparse
import jsonlines
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def get_parser_object_for_code_generation_script():
    """
    Returns the parser object for the code generation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_file_name",
        type=str,
        default="prompts.jsonl",
        help="Name of the prompts file in jsonl format.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Name of the model from the HuggingFace Transformer AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Name of the tokenizer in HuggingFace. Default is None (same as model_name).",
    )
    parser.add_argument(
        "--generation_strategy",
        type=str,
        default=None,
        help="HuggingFace generation config option. Default is None.",
    )
    parser.add_argument(
        "--stop_tokens",
        type=str,
        default=None,
        help="List of strings representing stop tokens. Default is None.",
    )
    parser.add_argument(
        "--prefix_instruction",
        type=str,
        default=None,
        help="String to be prepended to each prompt. Default is None.",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default=None,
        help="String of the output file name to save the results in jsonl format. Default is None.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="String of the device name. Default is 'cpu'.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="Name of the Weights & Biases project. Default is None.",
    )
    parser.add_argument(
        "--prompt_text_key",
        type=str,
        default="prompt",
        help="Name of the prompt text key in the prompts file. Default is 'prompt'.",
    )
    parser.add_argument(
        "--prompt_id_key",
        type=str,
        default="name",
        help="Name of the prompt ID key in the prompts file. Default is 'task_id'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for generating outputs. Default is 50.",
    )

    return parser


def initialize_wandb(model_name, wandb_project_name=None):
    '''
    Initializes Weights & Biases for logging the experiment parameters and outputs.

    Args:
        model_name (str): Name of the model from the HuggingFace Transformer AutoModelForCausalLM.
        wandb_project_name (str, optional): Name of the Weights & Biases project. Default is None.

    Returns:
        None
    '''
    
    # Initialize Weights & Biases
    if wandb_project_name == None:
        todays_date = datetime.today().strftime("%Y-%m-%d")
        wandb_project_name = f"inference-{model_name}-{todays_date}"

    # wanb project name cannot contain characters '/,\\,#,?,%,:'
    wandb_project_name = wandb_project_name.replace("/", "-")
    wandb.init(project=wandb_project_name)


def initialize_model_and_tokenizer(model_name, tokenizer_name=None, device="cpu"):
    """
    Initializes the model and tokenizer.

    Args:
        model_name (str): Name of the model from the HuggingFace Transformer AutoModelForCausalLM.
        tokenizer_name (str, optional): Name of the tokenizer in HuggingFace. Default is None (same as model_name).
        device (str, optional): String of the device name. Default is 'cpu'.

    Returns:
        tuple(AutoModelForCausalLM, AutoTokenizer): Tuple of the model and tokenizer.
    """
    
    if tokenizer_name == None:
        tokenizer_name = model_name
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def initialize_generation_strategy(generation_strategy_name):
    """
    Initializes the generation strategy.

    Args:
        generation_strategy_name (str, optional): HuggingFace generation config option. Default is None.

    Returns:
        GenerationConfig: HuggingFace generation config object.
    """ 
    if generation_strategy_name == None:
        generation_strategy = GenerationConfig(
            do_sample=True,
            top_p=0.95,
            temperature=0.2,
            num_return_sequences=1,
            max_new_tokens=50,
        )
    else:
        generation_strategy = GenerationConfig.from_pretrained(generation_strategy_name)

    return generation_strategy
        

def batched_sampling_from_model(model, inputs, generation_strategy, max_num_return_sequences):
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

    target_return_sequences = generation_strategy.num_return_sequences
    split_generations_into_batches = max_num_return_sequences < target_return_sequences
    if not split_generations_into_batches:
            outputs = model.generate(**inputs, generation_config=generation_strategy)
    else:
        # Split the generations into batches of 50
        outputs = []
        size_of_batch = max_num_return_sequences
        num_of_batches = target_return_sequences // size_of_batch
        size_of_last_batch = target_return_sequences % size_of_batch

        updated_generation_strategy = generation_strategy
        updated_generation_strategy.num_return_sequences = size_of_batch

        for i in tqdm(range(num_of_batches), unit="batch"):
            output = model.generate(**inputs, generation_config=updated_generation_strategy)
            outputs.append(output)
            if size_of_last_batch != 0 and i == num_of_batches-1:
                updated_generation_strategy.num_return_sequences = size_of_last_batch
                output = model.generate(**inputs, generation_config=updated_generation_strategy)
                outputs.append(output)

        #pad all the outputs to the length of the longest output
        max_length = max([output.shape[-1] for output in outputs])
        pad_token_id = generation_strategy.pad_token_id
        for i, output in enumerate(outputs):
            outputs[i] = torch.nn.functional.pad(output, (0, max_length - output.shape[-1]), mode='constant', value=pad_token_id)
        outputs = torch.cat(outputs, dim=0)
    
    return outputs

        
def store_output(generations, decoded_outputs, generation_strategy, prompt_id, prompt_text, tests, stop_tokens, lang, output_file_name):
    """
    Stores the output in the generations list.

    Args:
        generations (list[dict]): List of dictionaries containing prompt ID and corresponding generated output.
        decoded_outputs (list[str]): List of decoded outputs.
        generation_strategy (GenerationConfig): HuggingFace generation config object.
        prompt_id (int): Prompt ID.
        prompt_text (str): Prompt text.
        tests (str): Prompt tests.
        stop_tokens (list[str]): List of strings representing stop tokens.
        lang (str): Programming language.
        output_file_name (str): Name of the output file.


    Returns:
        list[dict]: List of dictionaries containing prompt ID and corresponding generated output.
    """
    for i, decoded_output in enumerate(decoded_outputs):
        generations.append(
            {
                "name": prompt_id,
                "language": lang,
                "temprature": generation_strategy.temperature,
                "top_p": generation_strategy.top_p,
                "max_new_tokens": generation_strategy.max_new_tokens,
                "prompt": prompt_text,
                "tests": tests,
                "stop_tokens": stop_tokens,
                "output_id": i,
                "output_text": decoded_output,
            }
        )

    if output_file_name == None:
        todays_date = datetime.today().strftime("%Y-%m-%d")
        output_file_name = f"{lang}-{todays_date}.jsonl"
    

    # if output_file_name exists, append the generations to the file, else create a new file
    if os.path.exists(output_file_name):
        with jsonlines.open(output_file_name, "a") as writer:
            for generation in generations:
                writer.write(generation)
    else:
        with jsonlines.open(output_file_name, "w") as writer:
            for generation in generations:
                writer.write(generation)


    return generations


def remove_prompt(outputs, inputs):
    """
    Removes the prompt from the outputs.

    Args:
        outputs (torch.Tensor): Generated output tensor.
        inputs (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Generated output tensor without the prompt.
    """
    prompt_length = inputs.shape[-1]
    return outputs[:, prompt_length:]


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


def read_prompts(prompts_file_name, prompt_text_key="prompt", prompt_id_key="name", prompt_test_key="tests"):
    """
    Reads the prompts from a jsonl file.

    Args:
        prompts_file_name (str): Name of the prompts file in jsonl
        format.
        prompt_text_key (str, optional): Name of the prompt text key in the prompts file. Default is 'prompt'.
        prompt_id_key (str, optional): Name of the prompt ID key in the prompts file. Default is 'task_id'.
        prompt_test_key (str, optional): Name of the prompt test key in the prompts file. Default is 'tests'.

    Returns:
        list[tuple(int, str, str)]: List of tuples containing prompt IDs , prompt texts and prompt tests.
    """
    prompts = []
    with jsonlines.open(prompts_file_name) as reader:
        for prompt in reader:
            prompt_id = prompt[prompt_id_key]
            prompt_text = prompt[prompt_text_key]
            prompt_test = prompt[prompt_test_key]
            prompts.append((prompt_id, prompt_text, prompt_test))
    return prompts


def generate_outputs(
    prompts,
    model_name,
    lang,
    tokenizer_name=None,
    generation_strategy_name=None,
    stop_tokens=None,
    prefix_instruction=None,
    output_file_name=None,
    device="cpu",
    wandb_project_name=None,
    max_num_return_sequences=50,
):
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

    # Initialize Weights & Biases
    initialize_wandb(model_name, wandb_project_name)


    # Load the model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(model_name, tokenizer_name, device)

    # List to store generations
    generations = []

    # Set the generation strategy
    generation_strategy = initialize_generation_strategy(generation_strategy_name)

    # Set the pad token id
    generation_strategy.pad_token_id = tokenizer.pad_token_id

    # Save the generation strategy
    generation_strategy.save_pretrained("generation_strategy_temp")

    # Log Experiment parameters
    wandb.config.update(
        {
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "do_sample": generation_strategy.do_sample,
            "top_p": generation_strategy.top_p,
            "temperature": generation_strategy.temperature,
            "num_return_sequences": generation_strategy.num_return_sequences,
            "max_new_tokens": generation_strategy.max_new_tokens,
            "pad_token_id": generation_strategy.pad_token_id,
            "stop_tokens": stop_tokens,
            "prefix_instruction": prefix_instruction,
            "output_file_name": output_file_name,
            "device": device,
        }
    )


    # Generate outputs for each prompt
    for prompt_id, prompt_text, tests in tqdm(prompts, unit="prompt"):
        start_time = datetime.now()
        generation_strategy = GenerationConfig.from_pretrained("generation_strategy_temp")

        # Prepend the prefix instruction to the prompt text
        if prefix_instruction:
            prompt_text = prefix_instruction + prompt_text

        # Encode the input text
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # Generate the output
        outputs = batched_sampling_from_model(model, inputs, generation_strategy, max_num_return_sequences)
        
        # Remove prompt from the output
        outputs = remove_prompt(outputs, inputs["input_ids"])

        # Decode the output
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Stop at stop tokens
        decoded_outputs = [stop_at_stop_token(decoded_output, stop_tokens) for decoded_output in decoded_outputs]

        # Store the output
        generations = store_output(generations, decoded_outputs, generation_strategy, prompt_id, prompt_text, tests, stop_tokens, lang, output_file_name)

        # Log generation time using wandb
        wandb.log({"generation_time": (datetime.now() - start_time).total_seconds()})


    # Save the outputs to a jsonl file
    if output_file_name == None:
        todays_date = datetime.today().strftime("%Y-%m-%d")
        output_file_name = f"{model_name}-{todays_date}.jsonl"
    with jsonlines.open(output_file_name, "w") as writer:
        for generation in generations:
            writer.write(generation)

    # Log outputs using Weights & Biases
    wandb.log({"generations": generations})

    return generations


def main():
    # Get inputs from arguments
    parser = get_parser_object_for_code_generation_script()
    args = parser.parse_args()

    # Read the prompts
    prompts = read_prompts(args.prompts_file_name, args.prompt_text_key, args.prompt_id_key)

    #Extract programming language from prompts file name (e.g. prompts_py.jsonl)
    lang = args.prompts_file_name.split("_")[1].split(".")[0]

    # Split the stop tokens
    if args.stop_tokens != None:
        stop_tokens = args.stop_tokens.split(",")
    else:
        with open('stop_tokens.pkl', 'rb') as f:
            stop_tokens = pickle.load(f)

    # Print the Arguments
    print("Starting the code generation with the following arguments:")
    print(args)

    # Generate outputs
    generated_outputs = generate_outputs(
        prompts,
        args.model_name,
        lang,
        args.tokenizer_name,
        args.generation_strategy,
        stop_tokens,
        args.prefix_instruction,
        args.output_file_name,
        args.device,
        args.wandb_project_name,
        args.batch_size,
    )

    # Print the generated outputs
    for generated_output in generated_outputs:
        print(generated_output)

if __name__ == "__main__":
    main()

    # Example command:
    # python code_generation.py --prompts_file_name "prompts.jsonl" --model_name "gpt2" --tokenizer_name "gpt2" --generation_strategy "top_p" --stop_tokens "['\n']" --prefix_instruction "def foo(x):\n\t" --output_file_name "generated_outputs.jsonl" --device "cpu" --wandb_project_name "inference-gpt2-2021-08-10" --prompt_text_key "prompt" --prompt_id_key "task_id"

    # First install the requirements using the following command:
    # pip install -r requirements.txt

    # Then login to Weights & Biases using the following command:
    # wandb login

    # Then login to HuggingFace using the following command:
    # transformers-cli login