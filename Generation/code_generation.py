import wandb
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
        default="task_id",
        help="Name of the prompt ID key in the prompts file. Default is 'task_id'.",
    )

    return parser

def generate_outputs(
    prompts,
    model_name,
    tokenizer_name=None,
    generation_strategy=None,
    stop_tokens=None,
    prefix_instruction=None,
    output_file_name=None,
    device="cpu",
    wandb_project_name=None,
):
    """
    Generates outputs based on the given prompts using a language model.

    Args:
        prompts (list[tuple(int, str)]): List of tuples containing prompt IDs and prompt texts.
        model_name (str): Name of the model from the HuggingFace Transformer AutoModelForCausalLM.
        tokenizer_name (str, optional): Name of the tokenizer in HuggingFace. Default is None (same as model_name).
        generation_strategy (str, optional): HuggingFace generation config option. Default is None.
        stop_tokens (list[str], optional): List of strings representing stop tokens. Default is None.
        prefix_instruction (str, optional): String to be prepended to each prompt. Default is None.
        output_file_name (str, optional): String of the output file name to save the results in jsonl format. Default is None.
        device (str, optional): String of the device name. Default is 'cpu'.

    Returns:
        list[dict]: List of dictionaries containing prompt ID and corresponding generated output.
    """

    # Initialize Weights & Biases
    if wandb_project_name == None:
        todays_date = datetime.today().strftime("%Y-%m-%d")
        wandb_project_name = f"inference-{model_name}-{todays_date}"
    wandb.init(project=wandb_project_name)

    # Load the model and tokenizer
    if tokenizer_name == None:
        tokenizer_name = model_name
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # List to store generations
    generations = []

    # Set the generation strategy
    if generation_strategy == None:
        generation_strategy = GenerationConfig(
            do_sample=True,
            top_p=0.95,
            temperature=0.2,
            num_return_sequences=1,
            max_new_tokens=50,
            pad_token_id=tokenizer.pad_token_id,
        )

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
    for prompt_id, prompt_text in tqdm(prompts):
        # track generation time using wandb
        start_time = datetime.now()

        # Prepend the prefix instruction to the prompt text
        if prefix_instruction:
            prompt_text = prefix_instruction + prompt_text

        # Encode the input text
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # Generate the output
        outputs = model.generate(**inputs, generation_config=generation_strategy)

        # Remove prompt from the output
        outputs = remove_prompt(outputs, inputs["input_ids"])

        # Decode the output
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Stop at stop tokens
        if stop_tokens != None:
            decoded_outputs = [
                stop_at_stop_token(decoded_output, stop_tokens)
                for decoded_output in decoded_outputs
            ]

        # Store the output
        for i, decoded_output in enumerate(decoded_outputs):
            generations.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "output_id": i,
                    "output_text": decoded_output,
                }
            )

        # Log generation time using wandb
        end_time = datetime.now()
        generation_time = end_time - start_time
        # convert to seconds
        generation_time = generation_time.total_seconds()
        wandb.log({"generation_time": generation_time})

        # Log the output using wandb
        wandb.log({"output": decoded_output})

        # Log the prompt using wandb
        wandb.log({"prompt": prompt_text})

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
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]

def read_prompts(prompts_file_name, prompt_text_key="prompt", prompt_id_key="task_id"):
    """
    Reads the prompts from a jsonl file.

    Args:
        prompts_file_name (str): Name of the prompts file in jsonl
        format.

    Returns:
        list[tuple(int, str)]: List of tuples containing prompt IDs and prompt texts.
    """
    prompts = []
    with jsonlines.open(prompts_file_name) as reader:
        for prompt in reader:
            prompt_id = prompt[prompt_id_key]
            prompt_text = prompt[prompt_text_key]
            prompts.append((prompt_id, prompt_text))
    return prompts

def main():
    # Get inputs from arguments
    parser = get_parser_object_for_code_generation_script()
    args = parser.parse_args()

    # Read the prompts
    prompts = read_prompts(args.prompts_file_name, args.prompt_text_key, args.prompt_id_key)

    # Print the Arguments
    print("Starting the code generation with the following arguments:")
    print(args)

    # Generate outputs
    generated_outputs = generate_outputs(
        prompts,
        args.model_name,
        args.tokenizer_name,
        args.generation_strategy,
        args.stop_tokens,
        args.prefix_instruction,
        args.output_file_name,
        args.device,
        args.wandb_project_name,
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