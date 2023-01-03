import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

''' Arguments '''
@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    valid_file: Optional[str] = field(default=None)
    test_file: Optional[str] = field(default=None)
    num_beams: Optional[int] = field(default=None)
    ignore_pad_token_for_loss: bool = field(default=True)
    source_prefix: Optional[str] = field(default="") # useful for T5
    forced_bos_token: Optional[str] = field(default=None) # useful for mBART (specify the language token)

@dataclass
class ModelArguments:
    model_name_or_path: str = field()

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

''' Logging '''
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
set_seed(training_args.seed)

raw_datasets = {"train": None, "valid": None, "test": None}

''' Model, Tokenizer '''
config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
model.resize_token_embeddings(len(tokenizer))

prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

''' Dataset '''
padding = False

def preprocess_function(examples):
    inputs, targets = [], []
    inputs = examples["inputs"]
    targets = examples["targets"]

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=1024, padding=padding, truncation=True)
    labels = tokenizer(text_target=targets, max_length=1024, padding=padding, truncation=True)
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if training_args.do_train:
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
    )

if training_args.do_eval:
    eval_dataset = raw_datasets["valid"]
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
    )

if training_args.do_predict:
    predict_dataset = raw_datasets["test"]
    predict_dataset = predict_dataset.map(
        preprocess_function,
        batched=True,
    )

# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 if training_args.fp16 else None,
)

''' Metric '''
metric = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

''' Trainer '''
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics if training_args.predict_with_generate else None,
)

''' Training'''
if training_args.do_train:
    train_result = trainer.train()
    metrics = train_result.metrics

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

''' Evaluation '''
if training_args.do_eval:
    logger.info("*** Evaluate ***")
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

''' Prediction '''
if training_args.do_predict:
    logger.info("*** Predict ***")

    predict_results = trainer.predict(
        predict_dataset, metric_key_prefix="predict", num_beams=num_beams
    )
    metrics = predict_results.metrics

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))

