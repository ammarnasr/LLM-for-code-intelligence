from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from finetuning_datasets import ConstantLengthDataset
import os
import numpy as np
from peft import PeftConfig, PeftModel

def main(ablation_var, ablation_values, start_index=0, continue_from_checkpoint=False):
    model_id = 'Salesforce/codegen-350M-mono'
    tokenizer_id = 'Salesforce/codegen-350M-mono'
    lang = 'java'
    dataset_id = f'ammarnasr/the-stack-{lang}-clean'
    effective_seq_length_train = 2048
    effective_seq_length_eval  = 2048
    lora_rank = 64
    lora_alpha = lora_rank*2
    lora_dropout = 0.05
    lora_bias = 'all'
    lora_task_type = 'CAUSAL_LM'
    lora_target_modules = ["qkv_proj", "out_proj", "lm_head", "fc_in", "fc_out"]
    dataloader_drop_last = True
    max_steps = 1000
    eval_steps = 50
    save_steps = 100
    eval_strategy = 'steps'
    logging_steps = 1
    learning_rate = 5e-5
    warmup_steps = 100
    lr_scheduler_type = 'cosine'
    gradient_checkpointing = True
    gradient_accumulation_steps = 1
    per_device_train_batch_size  = 4
    per_device_eval_batch_size = 4
    fp16 = True
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, use_cache=False)
    dataset = load_dataset(dataset_id)
    train_ds = dataset["train"]
    valid_ds = dataset["valid"]
    valid_ds = valid_ds.select(list(range(100)))
    
    def initilize_expiremnt(model, effective_seq_length_train, effective_seq_length_eval, lora_rank, lora_alpha, lora_dropout, lora_bias, lora_task_type, lora_target_modules):
        train_dataset = ConstantLengthDataset(tokenizer, train_ds, infinite=True, seq_length=effective_seq_length_train)
        valid_dataset = ConstantLengthDataset(tokenizer, valid_ds, infinite=False, seq_length=effective_seq_length_eval)
        lora_config = LoraConfig(r = lora_rank, lora_alpha = lora_alpha, lora_dropout = lora_dropout, bias = lora_bias, task_type = lora_task_type, target_modules = lora_target_modules)
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return train_dataset, valid_dataset, model
    
    def get_training_args_dict(output_dir, learning_rate, per_device_train_batch_size):
        training_args_dict = {}
        training_args_dict.update({
                "output_dir": output_dir,
                "run_name": output_dir+'-wandb',
                "dataloader_drop_last": dataloader_drop_last,
                "max_steps": max_steps,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "evaluation_strategy": eval_strategy,
                "logging_steps": logging_steps,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "lr_scheduler_type": lr_scheduler_type,
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": per_device_eval_batch_size,
                "fp16": fp16
        })
        return training_args_dict
    
    def get_model_from_checkpoint(output_dir):
        #Get all folders in output_dir that start with 'checkpoint-'
        checkpoint_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)) and f.startswith('checkpoint-')]
        #Get checkpoint indices
        checkpoint_indices = [int(f.split('-')[1]) for f in checkpoint_folders]
        #Get the checkpoint folder with the highest index
        latest_checkpoint_folder = checkpoint_folders[np.argmax(checkpoint_indices)]
        #Get the path to the latest checkpoint
        latest_checkpoint_path = os.path.join(output_dir, latest_checkpoint_folder)
        #Load the model from the latest checkpoint
        model_name = latest_checkpoint_path
        config = PeftConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,  trust_remote_code=True, revision="main")
        model = PeftModel.from_pretrained(model, model_name)
        print("Done loading the model from checkpoint: ", model_name, "With peft ...")
        model.print_trainable_parameters()
        return model
    

    if ablation_var == "lora_target_modules":
        for i in range(start_index, len(ablation_values)):
            lora_target_modules = ablation_values[i]
            train_dataset, valid_dataset, model = initilize_expiremnt(model, effective_seq_length_train, effective_seq_length_eval, lora_rank, lora_alpha, lora_dropout, lora_bias, lora_task_type, lora_target_modules)
            output_dir = f"codegen-{lang}-LoRa-v7-run-1-{ablation_var}-{i}-{lora_target_modules}"
            training_args_dict = get_training_args_dict(output_dir, learning_rate, per_device_train_batch_size)
            training_args = TrainingArguments(**training_args_dict)
            if continue_from_checkpoint:
                model = get_model_from_checkpoint(output_dir)
            trainer = Trainer(model, training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)
            trainer.train(resume_from_checkpoint=continue_from_checkpoint)
            
    elif ablation_var == "lora_rank":
        for i in range(start_index, len(ablation_values)):
            lora_rank = ablation_values[i]
            train_dataset, valid_dataset, model = initilize_expiremnt(model, effective_seq_length_train, effective_seq_length_eval, lora_rank, lora_alpha, lora_dropout, lora_bias, lora_task_type, lora_target_modules)
            output_dir = f"codegen-{lang}-LoRa-v7-run-1-{ablation_var}-{i}-{lora_rank}"
            training_args_dict = get_training_args_dict(output_dir, learning_rate, per_device_train_batch_size)
            training_args = TrainingArguments(**training_args_dict)
            trainer = Trainer(model, training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)
            trainer.train(resume_from_checkpoint=continue_from_checkpoint)

    elif ablation_var == "effective_seq_length_train":
        for i in range(start_index, len(ablation_values)):
            effective_seq_length_train = ablation_values[i]
            train_dataset, valid_dataset, model = initilize_expiremnt(model, effective_seq_length_train, effective_seq_length_eval, lora_rank, lora_alpha, lora_dropout, lora_bias, lora_task_type, lora_target_modules)
            output_dir = f"codegen-{lang}-LoRa-v7-run-1-{ablation_var}-{i}-{effective_seq_length_train}"
            training_args_dict = get_training_args_dict(output_dir, learning_rate, per_device_train_batch_size)
            training_args = TrainingArguments(**training_args_dict)
            if continue_from_checkpoint:
                model = get_model_from_checkpoint(output_dir)
            trainer = Trainer(model, training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)
            trainer.train(resume_from_checkpoint=continue_from_checkpoint)

    elif ablation_var == "per_device_train_batch_size":
        for i in range(start_index, len(ablation_values)):
            per_device_train_batch_size = ablation_values[i]
            train_dataset, valid_dataset, model = initilize_expiremnt(model, effective_seq_length_train, effective_seq_length_eval, lora_rank, lora_alpha, lora_dropout, lora_bias, lora_task_type, lora_target_modules)
            output_dir = f"codegen-{lang}-LoRa-v7-run-1-{ablation_var}-{i}-{per_device_train_batch_size}"
            training_args_dict = get_training_args_dict(output_dir, learning_rate, per_device_train_batch_size)
            training_args = TrainingArguments(**training_args_dict)
            if continue_from_checkpoint:
                model = get_model_from_checkpoint(output_dir)
            trainer = Trainer(model, training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)
            trainer.train(resume_from_checkpoint=continue_from_checkpoint)

    elif ablation_var == "learning_rate":
        for i in range(start_index, len(ablation_values)):
            learning_rate = ablation_values[i]
            train_dataset, valid_dataset, model = initilize_expiremnt(model, effective_seq_length_train, effective_seq_length_eval, lora_rank, lora_alpha, lora_dropout, lora_bias, lora_task_type, lora_target_modules)
            output_dir = f"codegen-{lang}-LoRa-v7-run-1-{ablation_var}-{i}-{learning_rate}"
            training_args_dict = get_training_args_dict(output_dir, learning_rate, per_device_train_batch_size)
            training_args = TrainingArguments(**training_args_dict)
            if continue_from_checkpoint:
                model = get_model_from_checkpoint(output_dir)
            trainer = Trainer(model, training_args, train_dataset=train_dataset, eval_dataset=valid_dataset)
            trainer.train(resume_from_checkpoint=continue_from_checkpoint)

    else:
        print("Invalid ablation_var")
        exit(1)





if __name__ == "__main__":
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
    learning_rate_list = [5e-6, 1e-5, 5e-5, 5e-4, 5e-3]

    # ablation_var = "lora_rank"
    # ablation_values = loRa_rank_list

    # ablation_var = "effective_seq_length_train"
    # ablation_values = effective_seq_length_list

    # ablation_var = "per_device_train_batch_size"
    # ablation_values = train_batch_size_list

    # ablation_var = "learning_rate"
    # ablation_values = learning_rate_list

    ablation_var = "lora_target_modules"
    ablation_values = LoRa_traget_module_list

    main(ablation_var, ablation_values, start_index=0, continue_from_checkpoint=None)

    