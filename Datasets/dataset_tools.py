from datasets import load_dataset
from tqdm.auto import tqdm
from datasets import DatasetDict

# "assembly", "batchfile", "c++", "c", "c-sharp", 
# "cmake", "css", "dockerfile", "visual-basic",
# "fortran", "go", "haskell", "html", "java",
# "javascript", "julia", "lua", "makefile", "markdown", 
# "perl", "php", "powershell", "python", "ruby", "rust", 
# "scala", "shell", "sql", "tex", "typescript"

all_languages = ["assembly", "batchfile", "c++", "c", "c-sharp",
"cmake", "css", "dockerfile", "visual-basic",
"fortran", "go", "haskell", "html", "java",
"javascript", "julia", "lua", "makefile", "markdown",
"perl", "php", "powershell", "python", "ruby", "rust",
"scala", "shell", "sql", "tex", "typescript"]


def downscale_dataset(dataset, size=1000000):
    '''
    Downscale the dataset to a smaller size (default: 1M)
    '''
    dataset = dataset.select(range(size))
    return dataset

def filter_dataset_columns(dataset, to_keep_columns=['hexsha', 'size', 'content', 'avg_line_length', 'max_line_length', 'alphanum_fraction']):
    '''
    Remove columns from the dataset except for:
    hexsha,	size, content, avg_line_length, max_line_length, alphanum_fraction
    '''
    all_columns = dataset.column_names
    columns_to_remove = [column for column in all_columns if column not in to_keep_columns]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset

def filter_dataset_rows(dataset, avg_line_length_upper_limit=100, max_line_length_upper_limit=1000, alphanum_fraction_lower_limit=0.25):
    '''
    Remove rows from the dataset that have:
    avg_line_length > 100
    max_line_length > 1000
    alphanum_fraction < 0.25
    '''
    indeices_to_remove = []
    tbar = tqdm(range(len(dataset)))
    processed = False
    for i in tbar:
        processed = False
        item = dataset[i]
        avg_line_length = item['avg_line_length']
        max_line_length = item['max_line_length']
        alphanum_fraction = item['alphanum_fraction']
        if avg_line_length > avg_line_length_upper_limit and not processed:
            processed = True
        if max_line_length > max_line_length_upper_limit and not processed:
            processed = True
        if alphanum_fraction < alphanum_fraction_lower_limit and not processed:
            processed = True
        if processed:
            indeices_to_remove.append(i)
    indeices_to_keep = [i for i in range(len(dataset)) if i not in indeices_to_remove]
    dataset_filtered = dataset.select(indeices_to_keep)
    return dataset_filtered

def split_dataset(dataset, splits=[0.9, 0.5]):
    '''
    Split the dataset into train, test, valid on the following splits:
    train: 90%
    test: 5%
    valid: 5%
    '''
    train_testvalid_precentage = splits[0]
    test_valid_precentage = splits[1]
    train_testvalid = dataset.train_test_split(test_size=1-train_testvalid_precentage, shuffle=True)
    train_dataset = train_testvalid['train']
    test_valid_dataset = train_testvalid['test']
    test_valid = test_valid_dataset.train_test_split(test_size=test_valid_precentage, shuffle=True)
    test_dataset = test_valid['train']
    valid_dataset = test_valid['test']
    return train_dataset, test_dataset, valid_dataset

def combine_datasets(datasets):
    '''
    Combine the datasets into a single DatasetDict
    '''
    tarin_test_valid_datasets = DatasetDict({
        'train': datasets[0],
        'test': datasets[1],
        'valid': datasets[2]
    })
    return tarin_test_valid_datasets

def push_dataset_to_hub(dataset, dataset_id):
    '''
    Push the dataset to the HF hub
    '''
    dataset.push_to_hub(dataset_id)
    print("Dataset pushed to HF hub as", dataset_id)


def main(lang):
    '''
    Main function: 
    1. Load the dataset
    2. Downscale the dataset
    3. Filter the dataset
    4. Split the dataset
    5. Combine the dataset
    6. Push the dataset to the HF hub
    '''
    tartget_dataset_id=f"ammarnasr/the-stack-{lang}-clean"
    source_dataset_id="bigcode/the-stack"
    source_dataset_data_dir=f"data/{lang}"
    print("Loading dataset...")
    source_ds = load_dataset(source_dataset_id, data_dir=source_dataset_data_dir, split="train")
    print("Downscaling dataset...")
    downscaled_dataset = downscale_dataset(source_ds)
    print("Filtering dataset columns...")
    filtered_columns_dataset = filter_dataset_columns(downscaled_dataset)
    print("Filtering dataset rows...")
    filtered_rows_dataset = filter_dataset_rows(filtered_columns_dataset)
    print("Splitting dataset...")
    train_ds, test_ds, valid_ds = split_dataset(filtered_rows_dataset)
    print("Combining dataset...")
    combined_dataset = combine_datasets([train_ds, test_ds, valid_ds])
    print("Pushing dataset to HF hub...")
    push_dataset_to_hub(combined_dataset, tartget_dataset_id)



