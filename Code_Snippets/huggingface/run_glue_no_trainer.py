import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from logging import get_logger
import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed
import evaluate
import transformers
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import wandb

logger = get_logger(__name__)

''' Arguments '''
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--task_name",type=str,default=None,)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--model_name_or_path",type=str,required=True,)
    parser.add_argument("--per_device_train_batch_size",type=int,default=8,)
    parser.add_argument("--per_device_eval_batch_size",type=int,default=8,)
    parser.add_argument("--learning_rate",type=float,default=5e-5,)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1)
    parser.add_argument("--lr_scheduler_type",type=SchedulerType,default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--checkpointing_steps",type=str,default=None,)
    args = parser.parse_args()
    return args

args = parse_args()

''' Logging '''
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
set_seed(args.seed)

raw_datasets = {"train": None, "valid": None, "test": None}
label_list = raw_datasets["train"].unique("label")
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)

''' Model, Tokenizer '''
config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {id: label for label, id in config.label2id.items()}

''' Dataset '''
padding = False
label_to_id = None

def preprocess_function(examples):
    result = tokenizer(examples['input'], padding=padding, max_length=1024, truncation=True)
    result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

processed_datasets = raw_datasets.map(preprocess_function,batched=True)
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["valid"]

for index in random.sample(range(len(train_dataset)), 3):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

# padded to max length in preprocess, then use default data collator, which does nothing
# unpadded in preprocess, then use DataCollatorWithPadding, which pads batches
# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer
if padding:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorWithPadding(tokenizer) 

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

''' Optimizer '''
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.max_train_steps,
)

checkpointing_steps = args.checkpointing_steps
if checkpointing_steps is not None and checkpointing_steps.isdigit():
    checkpointing_steps = int(checkpointing_steps)

metric = evaluate.load("accuracy")

# Train!
total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

''' Train '''
logger.info("***** Running training *****")
# Only show the progress bar once on each machine.
progress_bar = tqdm(range(args.max_train_steps))
completed_steps = 0

for epoch in range(0, args.num_train_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                        torch.save(
                            {'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, output_dir)

            if completed_steps >= args.max_train_steps:
                break
            
        ''' Eval '''
        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            references = batch["labels"]
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

        wandb.log(
            {
                "accuracy" if args.task_name is not None else "glue": eval_metric,
                "train_loss": total_loss.item() / len(train_dataloader),
                "epoch": epoch,
                "step": completed_steps,
            }
        )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                torch.save(
                    {'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, output_dir)

if args.output_dir is not None:
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)
