from datasets import load_dataset
import json


def convert_to_jsonl(ds, output_file_name):
    '''
    Converts a dataset to jsonl format
    args:
        ds: dataset
        output_file_name: name of the output file
    returns:
        None
    '''
    with open(output_file_name, 'w') as f:
        for i in range(len(ds)):
            f.write(json.dumps(ds[i]) + '\n')

def create_problems_jsonl(lang):
    '''
    Creates a jsonl file with the problems for the given language
    args:
        lang: language
    returns:
        None
    '''
    ds_name= "nuprl/MultiPL-E"
    ds_subset = f"humaneval-{lang}"
    ds_split = "test"

    ds = load_dataset(ds_name, ds_subset, split=ds_split)
    convert_to_jsonl(ds, f"humaneval_{lang}.jsonl")



if __name__ == "__main__":
    langs = ['py', 'java', 'js']
    for lang in langs:
        create_problems_jsonl(lang)
