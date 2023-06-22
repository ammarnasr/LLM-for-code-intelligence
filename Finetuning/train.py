import os
# from accelerate import Accelerator
from finetuning_datasets import create_datasets
from peft import LoraConfig, get_peft_model#, prepare_model_for_int8_training
from peft import PeftConfig, PeftModel
from finetuning_utils import get_args, SavePeftModelCallback, LoadBestPeftModelCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, logging, set_seed


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def run_training(args, train_data, val_data):
    print("Loading the model")
    if 'ammarnasr/' in args.model_path:    
        config = PeftConfig.from_pretrained(args)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            use_auth_token=True,
            use_cache=not args.no_gradient_checkpointing,
            # load_in_8bit=True,
            # device_map={"": Accelerator().process_index},
        )
        # Load the LoRA model
        model = PeftModel.from_pretrained(model, args.model_path)
        print_trainable_parameters(model)
    else:

        # disable caching mechanism when using gradient checkpointing
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            use_auth_token=True,
            use_cache=not args.no_gradient_checkpointing,
            # load_in_8bit=True,
            # device_map={"": Accelerator().process_index},
        )
        # model = prepare_model_for_int8_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules = ["c_proj", "c_attn", "q_attn"]
            target_modules = ["qkv_proj"]
        )

        if args.ft_type == "peft":
            print("Using PEFT")
            model = get_peft_model(model, lora_config)

        print_trainable_parameters(model)

        train_data.start_iteration = 0

    print("Starting main loop")
    if args.subset is None:
        args.subset = 'none'
    if 'ammarnasr/' in args.model_path:
        run_name = args.model_path.split('/')[-1]
    else:
        run_name = f"{args.model_path.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{args.subset.split('/')[-1]}_{args.split}_{args.ft_type}"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        # gradient_checkpointing=not args.no_gradient_checkpointing,
        # fp16=not args.no_fp16,
        # bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=run_name,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=f"ammarnasr/{run_name}"
        # ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback])

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)

def main_from_checkpoint(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training_from_checkpoint(args, train_dataset, eval_dataset)

def run_training_from_checkpoint(args, train_data, val_data):
    checkpoint_parent_dir = './checkpoints'
    all_checkpoints = os.listdir(checkpoint_parent_dir)
    #remove files that does not start with 'checkpoint-'
    all_checkpoints = [x for x in all_checkpoints if x.startswith('checkpoint-')]
    print(all_checkpoints)
    all_checkpoints_steps = [int(x.split('-')[1]) for x in all_checkpoints]
    index_of_max = all_checkpoints_steps.index(max(all_checkpoints_steps))
    checkpoint_dir = os.path.join(checkpoint_parent_dir, all_checkpoints[index_of_max])
    print("Loading the model from checkpoint: ", checkpoint_dir)
    config = PeftConfig.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        use_auth_token=True,
        use_cache=not args.no_gradient_checkpointing,
        # load_in_8bit=True,
        # device_map={"": Accelerator().process_index},
    )
    # Load the LoRA model
    model = PeftModel.from_pretrained(model, checkpoint_dir)

    print_trainable_parameters(model)

    # train_dataset.start_iteration = 0

    print("Starting main loop")
    if args.subset is None:
        args.subset = 'none'
    run_name = f"{args.model_path.split('/')[-1]}_{args.dataset_name.split('/')[-1]}_{args.subset.split('/')[-1]}_{args.split}_{args.ft_type}"


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        # gradient_checkpointing=not args.no_gradient_checkpointing,
        # fp16=not args.no_fp16,
        # bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=run_name,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=f"ammarnasr/{run_name}"
        # ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data, callbacks=[SavePeftModelCallback, LoadBestPeftModelCallback])

    print("Training from checkpoint...", checkpoint_dir, "...")
    trainer.train(checkpoint_dir)


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
    # main_from_checkpoint(args)