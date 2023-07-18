import os
import torch
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from tqdm.notebook import tqdm
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed
)
from finetuning_datasets import ConstantLengthDataset


def get_derived_variables(effective_seq_length, LoRa_rank, LoRa_traget_module_index, train_batch_size, gradient_accumulation_steps, using_LoRa):
    total_tokens = 1000000
    max_steps = total_tokens // effective_seq_length
    max_steps = max_steps // train_batch_size
    max_steps = max_steps // gradient_accumulation_steps

    actual_total_tokens = total_tokens
    if max_steps > 1000:
        max_steps = 1000
        actual_total_tokens = max_steps * effective_seq_length * train_batch_size * gradient_accumulation_steps


    eval_steps = (5/100) * max_steps
    eval_steps = int(eval_steps)
    save_steps = (10/100) * max_steps
    save_steps = int(save_steps)


    output_dir = f"seq_{effective_seq_length}_rank_{LoRa_rank}_module_{LoRa_traget_module_index}_bs_{train_batch_size}_gradacc_{gradient_accumulation_steps}_tokens_{actual_total_tokens}"
    #create output dir if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # create subdirs LoRa and Full if not exist
    if not os.path.exists(os.path.join(output_dir, "LoRa")):
        os.makedirs(os.path.join(output_dir, "LoRa"))
    if not os.path.exists(os.path.join(output_dir, "Full")):
        os.makedirs(os.path.join(output_dir, "Full"))

    if using_LoRa:
        output_dir = os.path.join(output_dir, "LoRa")
    run_name = output_dir.replace("\\", "_").replace("/", "_")
    return output_dir, run_name, max_steps, eval_steps, save_steps




def main(abilation_var, abilation_index):
    print('Step1: Loading Tokenizer and Dataset')
    model_id = "Salesforce/codegen-350M-mono"
    tokenizer_id = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    dataset_id = "ammarnasr/bigcode-the-stack-dedup-java-small-subset"
    dataset = load_dataset(dataset_id)
    train_ds = dataset["train"]
    valid_ds = dataset["valid"]
    valid_ds = valid_ds.select(list(range(100)))
    valid_dataset = ConstantLengthDataset(tokenizer, valid_ds, infinite=False, seq_length=1024)

    print('Step 2: Getting abilation vars')
    effective_seq_length_list = [128, 256, 512, 1024, 2048]
    loRa_rank_list = [8, 16, 32, 64, 128]
    LoRa_traget_module_list = [
        ["qkv_proj"], # index 0
        ["qkv_proj", "out_proj"], # index 1
        ["qkv_proj", "lm_head"], # index 2
        ["qkv_proj", "out_proj", "lm_head"], # index 3
        ["qkv_proj", "fc_in"], # index 4
        ["qkv_proj", "fc_out"], # index 5
        ["qkv_proj", "fc_in", "fc_out"], # index 6
        ["qkv_proj", "out_proj", "lm_head", "fc_in", "fc_out"], # index 7
    ]
    train_batch_size_list = [1, 2, 4, 8, 16]
    gradient_accumulation_steps_list = [1, 2, 4, 8, 16]
    learning_rate_list = [5e-5, 5e-4, 5e-3, 5e-2]
    using_LoRa_list = [True, False]
    effective_seq_length = 512
    loRa_rank = 16
    LoRa_traget_module_index = 7
    train_batch_size = 4
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    using_LoRa = True


    # Defualt Variables
    using_LoRa = True
    learning_rate = 5e-5
    gradient_accumulation_steps = 1
    loRa_rank = 16
    train_batch_size = 4
    effective_seq_length = 512
    LoRa_traget_module_index = 7

    if abilation_var == 'LoRa_traget_module_index':
        LoRa_traget_module_index = abilation_index
    elif abilation_var == 'effective_seq_length':
        effective_seq_length = effective_seq_length_list[abilation_index]
    elif abilation_var == 'loRa_rank':
        loRa_rank = loRa_rank_list[abilation_index]
    elif abilation_var == 'train_batch_size':
        train_batch_size = train_batch_size_list[abilation_index]
    elif abilation_var == 'gradient_accumulation_steps':
        gradient_accumulation_steps = gradient_accumulation_steps_list[abilation_index]
    elif abilation_var == 'learning_rate':
        learning_rate = learning_rate_list[abilation_index]
    elif abilation_var == 'using_LoRa':
        using_LoRa = using_LoRa_list[abilation_index]
    else:
        print('Wrong abilation_var')
        print(f'''
                    Select One of:
              1. LoRa_traget_module_index, max value is {len(LoRa_traget_module_list)-1}
                2. effective_seq_length, max value is {len(effective_seq_length_list)-1}
                3. loRa_rank, max value is {len(loRa_rank_list)-1}
                4. train_batch_size, max value is {len(train_batch_size_list)-1}
                5. gradient_accumulation_steps, max value is {len(gradient_accumulation_steps_list)-1}
                6. learning_rate, max value is {len(learning_rate_list)-1}
                7. using_LoRa, max value is {len(using_LoRa_list)-1}    
              ''')
        
        return
    output_dir, run_name, max_steps, eval_steps, save_steps = get_derived_variables(effective_seq_length, loRa_rank, LoRa_traget_module_index, train_batch_size, gradient_accumulation_steps, using_LoRa)
    train_dataset = ConstantLengthDataset(tokenizer, train_ds, infinite=True, seq_length=effective_seq_length)
    lora_config = LoraConfig(r = loRa_rank, lora_alpha=loRa_rank*2, lora_dropout= 0.05, bias="all", task_type="CAUSAL_LM", target_modules = LoRa_traget_module_list[LoRa_traget_module_index])
    if using_LoRa:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, use_cache=False)
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    training_args_dict = {}
    training_args_dict.update({
            "output_dir": output_dir,
            "run_name": run_name,
            "max_steps": max_steps,
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": train_batch_size ,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "logging_steps": 1,
            "per_device_eval_batch_size": 4,
            "dataloader_drop_last": True,
            "evaluation_strategy": "steps",
            "gradient_checkpointing": True,
            "fp16": True,
    })
    print(training_args_dict)
    training_args = TrainingArguments(**training_args_dict)

    trainer = Trainer(model, training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)
    trainer.train()
    print("=====================================================")


