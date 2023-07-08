from datasets import load_dataset
from tqdm.auto import tqdm
from datasets import concatenate_datasets
from datasets import DatasetDict


def push_to_hf_datasets():
    dataset_id = "ammarnasr/bigcode-the-stack-dedup-java-small-subset"
    dataset = load_dataset(dataset_id)
    print("Dataset: ", dataset_id)
    print(dataset)
    train_ds = dataset["train"]
    valid_ds = dataset["test"]
    train_ds = concatenate_datasets([train_ds, valid_ds])

    train_testvalid = train_ds.train_test_split(test_size=0.1, shuffle=True)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.01, shuffle=True)
    final_train = train_testvalid['train']
    final_test = test_valid['train']
    final_valid = test_valid['test']
    print("final_train length: ", len(final_train))
    print("final_test length: ", len(final_test))
    print("final_valid length: ", len(final_valid))

    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': final_train,
        'test': final_test,
        'valid': final_valid
    })
    print("train_test_valid_dataset: ")
    print(train_test_valid_dataset)

    #Push to HF hub
    train_test_valid_dataset.push_to_hub(dataset_id)


def filter_dataset(dataset):
    indeices_to_remove = []
    seen_hashes = set()
    tbar = tqdm(range(len(dataset)))
    avg_line_length_removed = 0
    max_line_length_removed = 0
    alphanum_fraction_removed = 0
    processed = False
    for i in tbar:
        processed = False
        item = dataset[i]
        avg_line_length = item['avg_line_length']
        max_line_length = item['max_line_length']
        alphanum_fraction = item['alphanum_fraction']

        if avg_line_length > 100 and not processed:
            avg_line_length_removed += 1
            processed = True
        if max_line_length > 1000 and not processed:
            max_line_length_removed += 1
            processed = True
        if alphanum_fraction < 0.25 and not processed:
            alphanum_fraction_removed += 1
            processed = True
        if processed:
            indeices_to_remove.append(i)


        tbar.set_description(f"avg_line_length_removed: {avg_line_length_removed}, max_line_length_removed: {max_line_length_removed}, alphanum_fraction_removed: {alphanum_fraction_removed}")
        
            
        
        #Remove the items
        indeices_to_keep = [i for i in range(len(dataset)) if i not in indeices_to_remove]
        dataset_filtered = dataset.select(indeices_to_keep)
        return dataset_filtered
