import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import numpy as np
from datasets import load_dataset
import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
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
label_list = raw_datasets["train"].unique("label")
label_list.sort()  # Let's sort it for determinism
num_labels = len(label_list)

''' Model, Tokenizer '''
config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {id: label for label, id in config.label2id.items()}

''' Dataset '''
padding = False
label_to_id = None

def preprocess_function(examples):
    result = tokenizer(examples['input'], padding=padding, max_length=1024, truncation=True)
    result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
)

if training_args.do_train:
    train_dataset = raw_datasets["train"]
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

if training_args.do_eval:
    eval_dataset = raw_datasets["valid"]

if training_args.do_predict:
    predict_dataset = raw_datasets["test"]

# padded to max length in preprocess, then use default data collator, which does nothing
# unpadded in preprocess, then use DataCollatorWithPadding, which pads batches
# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer
if padding:
    data_collator = default_data_collator
else:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None) 

''' Metric '''
def compute_metrics(p: EvalPrediction):
    # metric = evaluate.load("glue", data_args.task_name)
    metric = evaluate.load("accuracy")
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    return result

''' Trainer '''
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
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
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

''' Prediction '''
if training_args.do_predict:
    logger.info("*** Predict ***")
    predict_dataset = predict_dataset.remove_columns("label")
    predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    predictions = np.argmax(predictions, axis=1)
    output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.task_name}.txt")
    if trainer.is_world_process_zero():
        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results {data_args.task_name} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                item = label_list[item]
                writer.write(f"{index}\t{item}\n")
