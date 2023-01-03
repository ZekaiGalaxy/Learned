def parse_args()

## load dataset, model, tokenizer
train_dataset, test_dataset, tokenizer, model

## process dataset
def preprocess_function(examples):
    # T5
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["original_text"]] 
    # Bart
    model_inputs = tokenizer(examples["inputs"], max_length=1024, truncation=True) 
    labels = tokenizer(examples["labels"], max_length=1024, truncation=True) 
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

## compute_metrics
rouge = load_metric("rouge")
bleu = load_metric("sacrebleu") 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
    # ROUGE
    decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]    
    rouge_result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
    result = {key: value.mid.fmeasure for key, value in rouge_result.items()}
        
    # Length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    # BLEU
    decoded_labels_expanded = [[x] for x in decoded_labels]
    result2 = bleu.compute(predictions=decoded_preds, references=decoded_labels_expanded)
    result['bleu'] = round(result2["score"], 1)
        
    return {k: round(v, 4) for k, v in result.items()}

## train
batch_size = 6
args = Seq2SeqTrainingArguments(
    "test-summarization",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train_datasets["train"],
    eval_dataset=tokenized_dev_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model("output/modelname")

## load & eval
model = AutoModelForSeq2SeqLM.from_pretrained("output/modelname")
tokenizer = AutoTokenizer.from_pretrained("output/modelname")
trained_model = pipeline('summarization', model=model, tokenizer=tokenizer)

texts = test_data['input'].to_list()
generated = [trained_model(phrase)[0]['summary_text'] for phrase in texts]

with open(os.path.join(path,'filename'), 'w') as f:
    for item in generated:
        f.write("%s\n" % item)


#######
For GPT, GPT2, refer to the following link.
https://github.com/SALT-NLP/positive-frames/blob/main/run.py
