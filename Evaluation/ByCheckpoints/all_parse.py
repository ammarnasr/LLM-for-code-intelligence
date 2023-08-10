import argparse

def get_parser_object_for_code_generation_script():
    """
    Returns the parser object for the code generation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_file_name",
        type=str,
        # default="humaneval_java.jsonl",
        default="Generation/humaneval_python.jsonl",
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
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for generating outputs. Default is 50.",
    )

    parser.add_argument(
        "--saved_model_path",
        type=str,
        default=None,
        help="Path to the saved model.",
    )

    return parser

