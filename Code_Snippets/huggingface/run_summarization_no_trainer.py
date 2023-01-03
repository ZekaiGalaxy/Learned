import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from transformers import set_seed
import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from logging import get_logger
import evaluate
import transformers
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

def pad(seq, dim, pad_index):
    sizes = [s.shape for s in seq]
    max_size = max(s[dim] for s in sizes)
    padded = []
    for s in seq:
        if s.shape[dim] == max_size:
            padded.append(s)
        else:
            pad_size = list(s.shape)
            pad_size[dim] = max_size - s.shape[dim]
            padded.append(torch.cat([s, torch.full(tuple(pad_size), pad_index, dtype=s.dtype, device=s.device)], dim=dim))
    return padded   

logger = get_logger(__name__)

''' Arguments '''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",type=str,default=None)
    parser.add_argument("--dataset_name",type=str,default=None)
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

    parser.add_argument("--ignore_pad_token_for_loss",type=bool,default=True)
    parser.add_argument("--source_prefix",type=str,default=None)
    parser.add_argument("--num_beams",type=int,default=None)
    args = parser.parse_args()
    return args

args = parse_args()

''' Logging '''
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
if args.seed is not None:
    set_seed(args.seed)

raw_datasets = {"train": None, "valid": None, "test": None}

''' Model, Tokenizer '''
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
model.resize_token_embeddings(len(tokenizer))

prefix = args.source_prefix if args.source_prefix is not None else ""

''' Dataset '''
padding = False

def preprocess_function(examples):
    inputs, targets = [], []
    inputs = examples["inputs"]
    targets = examples["targets"]

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)
    labels = tokenizer(text_target=targets, max_length=1024, padding=padding, truncation=True)
    if padding == "max_length" and args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

processed_datasets = raw_datasets.map(preprocess_function,batched=True)
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"]

for index in random.sample(range(len(train_dataset)), 1):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(tokenizer,model=model,
    label_pad_token_id=label_pad_token_id,
)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


''' Optimizer '''
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
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

metric = evaluate.load("rouge")

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
        gen_kwargs = {
            "max_length": args.val_max_target_length if args is not None else config.max_length,
            "num_beams": args.num_beams,
        }
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = pad(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                labels = batch["labels"]
                if not padding:
                    labels = pad(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                # If we are in a multiprocess environment, the last batch has duplicates
                if step == len(eval_dataloader) - 1:
                    decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                    decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += len(decoded_labels)

                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}

        logger.info(result)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                torch.save(
                    {'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, output_dir)

 
with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
    json.dump(
            {
                "eval_rouge1": result["rouge1"],
                "eval_rouge2": result["rouge2"],
                "eval_rougeL": result["rougeL"],
                "eval_rougeLsum": result["rougeLsum"],
            },
            f,
        )
 
