import os
import json
import gzip
import argparse
import jsonlines
from tqdm import tqdm

def gunzip_json(path):
    """
    Reads a .json.gz file, and produces None if any error occured.
    """
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        return None
    
def gzip_json(path, data):
    """
    Writes a .json.gz file, and produces None if any error occured.
    """
    try:
        with gzip.open(path, "wt") as f:
            json.dump(data, f)
    except Exception as e:
        return None
    
def convert_jsonl_structure(source_file, target_dir=None):
    """
    Convert the structure of a JSONL file from the first format to the second JSON format.

    Args:
        source_file (str): Path to the source JSONL file.
        target_dir (str): Path to the target directory. If None, the target directory is the same as the source directory.

    Returns:
        None
    """
    with jsonlines.open(source_file, 'r') as source_reader:
        source_data = list(source_reader)

    target_data = {}
    for json_data in tqdm(source_data, unit='jsonline'):
        completion = json_data['output_text']
        prompt = json_data['prompt']
        name = json_data['name']
        language = json_data['language']
        temprature = json_data['temprature']
        top_p = json_data['top_p']
        max_new_tokens = json_data['max_new_tokens']
        tests = json_data['tests']
        stop_tokens = json_data['stop_tokens']

        if name not in target_data:
            target_data[name] = {
                'prompt': prompt,
                'language': language,
                'temprature': temprature,
                'top_p': top_p,
                'max_new_tokens': max_new_tokens,
                'tests': tests,
                'stop_tokens': stop_tokens,
                'completions': [completion]
            }
        else:
            target_data[name]['completions'].append(completion)

    for name in target_data:
        if target_dir is None:
            target_file_name = f'{name}.json.gz'
        else:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            target_file_name = f'{target_dir}/{name}.json.gz'
        gzip_json(target_file_name, target_data[name])

# Usage example
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert the structure of a JSONL file from the first format to the second JSON format.')
    parser.add_argument('--source_file', type=str, required=True, help='Path to the source JSONL file.')
    parser.add_argument('--target_dir', type=str, default=None, help='Path to the target directory. If None, the target directory is the same as the source directory.')
    args = parser.parse_args()

    # Convert the structure of the JSONL file
    convert_jsonl_structure(args.source_file, args.target_dir)
    print('Done.')
