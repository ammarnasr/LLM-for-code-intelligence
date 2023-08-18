from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from finetuning_datasets import ConstantLengthDataset
import os
import numpy as np
from peft import PeftConfig, PeftModel
import pandas as pd
import numpy as np               
import matplotlib.pyplot as plt  
plt.style.use('bmh')



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
        print("Loading the model from checkpoint: ", model_name, "With peft ...")
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

    


def read_csv_from_dir(variable_dir, index, type, drop_min_max=False):
    csv_path = f'./{variable_dir}/{type}_{index}.csv'
 
    df = pd.read_csv(csv_path)
 
    if type == 'loss':
        new_cols = ['step', 'loss', 'loss_MIN', 'loss_MAX']
        new_cols = [f'{variable_dir}_{index}_{col}' for col in new_cols]
        df.columns = new_cols
        if drop_min_max:
            df = df.drop(columns=new_cols[2:])
    elif type == 'mem':
        new_cols = ['step', 'runtime', 'runtime_MIN', 'runtime_max', 'bytes', 'bytes_MIN', 'bytes_MAX']
        new_cols = [f'{variable_dir}_{index}_{col}' for col in new_cols]
        df.columns = new_cols
        if drop_min_max:
            df = df.drop(columns=new_cols[2:4] + new_cols[5:])

 
    return df

def read_indices_from_dir(variable_dir, type, indices, drop_min_max=False):
    dfs = []
    for index in indices:
        df = read_csv_from_dir(variable_dir, index, type, drop_min_max)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    columns = df.columns
    steps_cols = [col for col in columns if 'step' in col]
    if type == 'loss':
        first_steps = df[steps_cols[0]]
        same_steps = True
        for steps_col in steps_cols[1:]:
            steps = df[steps_col]
            if not steps.equals(first_steps):
                same_steps = False
                break
        if same_steps:
            df = df.drop(columns=steps_cols[1:])
            df = df.rename(columns={steps_cols[0]: 'step'})
    elif type == 'mem':
        df = df.drop(columns=steps_cols)
        data = {'runtime': [], 'bytes': [], 'model': []}
        columns = df.columns
        for col in columns:
            key = col.split('_')[-1]
            index = col.split('_')[-2]
            data[key].append(df[col].values[0])
            if key == 'bytes':
                data['model'].append(f'{variable_dir}_{index}')
        df = pd.DataFrame(data)
            
    return df

def plot_memory_time_bars(df, fig, ax1, variable_dir='Ablation Variable' ,width=0.35, c1='rebeccapurple', c2='hotpink',
                          memory_lim=15, time_lim=60, legend_labels=['Memory', 'Runtime']):
    ax2 = ax1.twinx()

    ind = np.arange(len(df)) 
    ax1.set_xticks(ind)
    ax1.set_xticklabels(df['model'])
    ax1.set_xlabel(variable_dir)
    
    rects1 = ax1.bar(ind - width/2, df['bytes'], width, label='Memory', color=c1)
    rects2 = ax2.bar(ind + width/2, df['runtime'], width, label='Runtime', color=c2)
    ax1.set_ylabel('Memory in GB')
    ax2.set_ylabel('Runtime in Minutes')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.yaxis.label.set_color(c1)
    ax2.yaxis.label.set_color(c2)
    c1 = 'black'
    c2 = 'black'
    ax1.tick_params(axis='y', colors=c1)
    ax2.tick_params(axis='y', colors=c2)
    ax1.legend(lines + lines2, legend_labels, loc='upper left')
    ax1.set_ylim(0, memory_lim)
    ax2.set_ylim(0, time_lim)
    ax1.grid(False)
    ax2.grid(False)

    title = f'Memory and Runtime for Different {variable_dir} values'
    plt.title(title)
    fig.tight_layout()
    plt.show()
    return fig, ax1, ax2

def plot_loss_line_plot(df, fig, ax1, variable_dir='Ablation Variable',
                        width=0.35, font_size=24, save_path=None,
                        legend_labels=None, title=None):
    #Line plot with x axis as step and multiple lines for each model


    x = df['step']
    columns = df.columns
    columns = [col for col in columns if 'loss' in col]
    for col in columns:
        y = df[col]
        ax1.plot(x, y, label=col, linewidth=3)

    ax1.set_xlabel('Step', fontsize = font_size)
    ax1.set_ylabel('Loss', fontsize = font_size)
    if title is None:
        ax1.set_title("Loss vs. " + variable_dir, fontsize = font_size)
    else:
        ax1.set_title(title, fontsize = font_size)
    if legend_labels is None:
        ax1.legend(fontsize = font_size)
    else:
        ax1.legend(legend_labels, fontsize = font_size)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = font_size)
    ax1.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.1, axis='y', zorder = 0)
    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle='-', linewidth='0.5', color='black', alpha=0.01, axis='y', zorder = 0)
    ax1.set_facecolor('#C7C7CF')
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    return fig, ax1


def plot_memory_runtime_line_plot(df, fig, ax1, ax2, variable_dir='Ablation Variable',
                        width=0.35, font_size=24, save_path=None,
                        legend_labels=None, title=None):
     #2 Line plots (1 plot with mirroed axis) with x axis as step and multiple lines for each model

    x = df['model']
    if legend_labels is None:
        x_ticks = [f'{variable_dir}_{i}' for i in range(len(x))]
    else:
        x_ticks = legend_labels
    columns = df.columns
    columns = [col for col in columns if 'runtime' in col]
    for col in columns:
        y = df[col]
        ax2.plot(x, y, label=col, linewidth=3, color='hotpink')

    columns = df.columns
    columns = [col for col in columns if 'bytes' in col]
    for col in columns:
        y = df[col]
        ax1.plot(x, y, label=col, linewidth=3, color='rebeccapurple')

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_ticks, rotation=0, ha='right')
    ax1.set_xlabel(variable_dir, fontsize = font_size)

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_ticks, rotation=0, ha='right')
    ax2.set_xlabel(variable_dir, fontsize = font_size)

    #combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.yaxis.label.set_color('rebeccapurple')
    ax2.yaxis.label.set_color('hotpink')
    ax1.tick_params(axis='y', colors='rebeccapurple')
    ax2.tick_params(axis='y', colors='hotpink')
    ax1.legend(lines + lines2, legend_labels, loc='upper left', fontsize = font_size)
    ax1.set_ylim(0, df['bytes'].max()*1.25)
    ax2.set_ylim(0, df['runtime'].max()*1.25)
    ax1.grid(False)
    ax2.grid(False)

    

    ax1.set_ylabel('Memory in GB', fontsize = font_size)
    ax2.set_ylabel('Runtime in Minutes', fontsize = font_size)
    if title is None:
        ax1.set_title("Memory Runtime vs. " + variable_dir, fontsize = font_size)
    else:
        ax1.set_title(title, fontsize = font_size)

    ax1.tick_params(axis = 'both', which = 'major', labelsize = font_size)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = font_size)
    ax1.grid(which='major', linestyle='-', linewidth='0.05', color='black', alpha=0.1, axis='y', zorder = 0)
    ax2.grid(which='major', linestyle='-', linewidth='0.05', color='black', alpha=0.1, axis='y', zorder = 0)
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax1.grid(which='minor', linestyle='-', linewidth='0.01', color='black', alpha=0.01, axis='y', zorder = 0)
    ax2.grid(which='minor', linestyle='-', linewidth='0.01', color='black', alpha=0.01, axis='y', zorder = 0)
    # ax1.set_facecolor('#C7C7CF')
    # ax2.set_facecolor('#C7C7CF')
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    return fig, ax1, ax2


