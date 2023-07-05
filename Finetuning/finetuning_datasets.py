import os
import json
import torch
import pickle
import jsonlines
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset
from datasets import load_dataset




def prepare_sample_text(example, input_column_name="prompt", output_column_name="completion"):
    """Prepare the text from a sample of the dataset."""
    if input_column_name == "content":
        text = example[input_column_name]
        text = text[:2000]
    else:
        text = f"Question: {example[input_column_name]}\n\nAnswer: {example[output_column_name]}"
    return text




def chars_token_ratio(dataset, tokenizer, input_column_name="prompt", output_column_name="completion", nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, input_column_name, output_column_name)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        input_column_name="prompt",
        output_column_name="completion"
    ):
        self.tokenizer = tokenizer
        # self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else args.eos_token_id
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(prepare_sample_text(next(iterator), self.input_column_name, self.output_column_name))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.input_column_name, args.output_column_name)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
        input_column_name=args.input_column_name,
        output_column_name=args.output_column_name
    )
    return train_dataset, valid_dataset



def convert_json_to_jsonlines(filename):
    with open(filename) as f:
        data = json.load(f)
    filename_without_ext = filename[:-5]
    jsonl_filename = filename_without_ext+'.jsonl'
    with jsonlines.open(jsonl_filename, mode='w') as writer:
        for i in tqdm(range(len(data['hexsha']))):
            writer.write({
                'hexsha': data['hexsha'][i],
                'size': data['size'][i],
                'content': data['content'][i],
                'avg_line_length': data['avg_line_length'][i],
                'max_line_length': data['max_line_length'][i],
                'alphanum_fraction': data['alphanum_fraction'][i]
            })
    return jsonl_filename

def push_to_huggingface(num_samples=1000000):
    print('Loading dataset from pickle file')
    trainfile = './data_pkl/java/bigcode-the-stack-dedup-train.pkl'
    with open(trainfile, "rb") as f:
        train_ds = pickle.load(f)
    
    train_ds_dict = {
        'hexsha': [],
        'size': [],
        'content': [],
        'avg_line_length': [],
        'max_line_length': [],
        'alphanum_fraction': []
    }
    save_every = num_samples//10
    trainfile_json = f'./data_pkl/java/bigcode-the-stack-dedup-train.json'
    length = len(train_ds['hexsha'])

    print(f'Creating json file with {num_samples} samples which is {num_samples/length*100}% of the dataset')
    current_size = 0

    tbar = tqdm(range(num_samples),total=num_samples, desc=f'latest file size: {current_size}MB')

    for i in tbar:
        train_ds_dict['hexsha'].append(train_ds[i]['hexsha'])
        train_ds_dict['size'].append(train_ds[i]['size'])
        train_ds_dict['content'].append(train_ds[i]['content'])
        train_ds_dict['avg_line_length'].append(train_ds[i]['avg_line_length'])
        train_ds_dict['max_line_length'].append(train_ds[i]['max_line_length'])
        train_ds_dict['alphanum_fraction'].append(train_ds[i]['alphanum_fraction'])
        if i % save_every == 0:
            with open(trainfile_json, 'w') as f:
                json.dump(train_ds_dict, f)
            current_size = os.path.getsize(trainfile_json)/1e6
            tbar.set_description(f'latest file size: {current_size:.2f}MB')
            

    print('Converting json to jsonl')
    trainfile_jsonl = convert_json_to_jsonlines(trainfile_json)

    print('Pushing to huggingface')
    hf_ds = load_dataset("json", data_files=trainfile_jsonl)
    hf_repo = 'ammarnasr/bigcode-the-stack-dedup-java-small-subset'
    hf_ds.push_to_hub(hf_repo)



class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=2.95,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 49152
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = "content"

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(input_ids),
                        "labels": torch.LongTensor(input_ids),
                    }



