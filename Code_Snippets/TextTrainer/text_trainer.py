import csv
from tqdm import tqdm
import torch
import evaluate
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup, set_seed
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader
from params import *
import os
from datasets import Dataset as HDataset
from datasets import DatasetDict
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from accelerate import DistributedDataParallelKwargs
import time

sacrebleu = evaluate.load('sacrebleu')
rouge = evaluate.load('rouge')
bertscore = evaluate.load("bertscore")

class TextDataset(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

class ToolBox:
    def save_txt(self, data, path, result=None):
        with open(path, 'w+') as f:
            if result is not None:
                print(result, file=f)
            for d in data:
                for dd in d:
                    f.write(dd)
                    f.write('\t')
                f.write('\n')

    def load_txt(self, path):
        data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                data.append(l.strip().split('\t'))
        return data

    def write_csv(self, data, file_name):
        with open(file_name, 'a', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data)

    def gettime(self):
        import  datetime,time
        return datetime.datetime.now().strftime('%Y%m%d')

    def save_json(self, args, path):
        import json
        with open(path, 'w') as f:
            json.dump(args.__dict__, f)

    def load_json(self, path):
        import json
        with open(path, 'r') as f:
            return json.load(f)
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

class TextTrainer(ToolBox):
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.test_loader = None
        self.args = None
        self.device = None
        self.train_data = None
        self.test_data = None
        self.total_step = 0
        self.modelname = None
        self.csv_path = None
        self.ddp = 0
    
    def report_result(self, write_data):
        self.write_csv(write_data, self.csv_path)
    
    def prepare_loader(self):
        assert self.train_data is not None
        assert self.test_data is not None
        train_dataset = TextDataset(self.train_data)
        test_dataset = TextDataset(self.test_data)
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        context, response=zip(*batch)
        context,response=list(context),list(response)
        context_ids,context_mask=self.tokenize_batch(context)
        response_ids,response_mask=self.tokenize_batch(response)
        return [context_ids,context_mask,response_ids,response_mask]
    
    def tokenize_batch(self, data):
        tokenized=self.tokenizer(data,padding=True,truncation=True,max_length=64)
        input_ids=torch.tensor(tokenized['input_ids'])
        attention_mask=torch.tensor(tokenized['attention_mask'])
        return input_ids,attention_mask

    def prepare_model(self):
        pass

    def prepare_optimizer(self):
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(self.train_loader)*self.args.n_epoch)

    def get_train_loss(self, data):
        context_ids, context_mask, response_ids, response_mask = [d.to(self.device) for d in data]
        response_ids[response_mask == 0] = -100
        input = {
            "input_ids": context_ids,
            "attention_mask": context_mask,
            "labels": response_ids
        }
        output = self.model(**input)
        loss = output.loss
        return loss
    
    def get_eval_input(self, data):
        if self.ddp == 1:
            context_ids, context_mask, response_ids, response_mask = data
        else:
            context_ids, context_mask, response_ids, response_mask = [d.to(self.device) for d in data]
        input = {
            "input_ids": context_ids,
            "attention_mask": context_mask,
        }
        labels = response_ids
        context = context_ids
        return input, context, labels
    
    def train_step(self, epoch):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader),total=len(self.train_loader))
        for idx, data in pbar:
            self.optimizer.zero_grad()
            loss = self.get_train_loss(data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        
            if self.args.eval_strategy == 'step' and self.total_step % self.args.eval_step == 0:
                self.valid_step(self.total_step)
                self.model.train()
            
            pbar.set_description('Step: %d \t| Loss: %.3f' % (idx, loss))
            pbar.update(1)
            self.total_step += 1
    
    def accelerate_train_step(self, epoch):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader),total=len(self.train_loader))
        for idx, data in pbar:
            self.optimizer.zero_grad()
            loss = self.get_train_loss(data)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
        
            if self.args.eval_strategy == 'step' and self.total_step % self.args.eval_step == 0 and self.accelerator.is_main_process:
                self.accelerate_valid_step(self.total_step)
                self.model.train()
            
            if self.accelerator.sync_gradients:
                pbar.set_description('Step: %d \t| Loss: %.3f' % (idx, loss))
                pbar.update(1)
                self.total_step += 1
    
    @torch.no_grad()
    def valid_step(self, num):
        self.model.eval()
        pbar = tqdm(enumerate(self.test_loader),total=len(self.test_loader))
        refs = []
        preds = []
        cons = []
        for idx, data in pbar:
            input, context, labels = self.get_eval_input(data)
            output = self.model.generate(**input, max_length=40, num_beams=4, early_stopping=True)
            output = output.to('cpu').tolist()
            for i in range(len(output)):
                cons.append(self.tokenizer.decode(context[i], skip_special_tokens=True))
                preds.append(self.tokenizer.decode(output[i], skip_special_tokens=True))
                refs.append(self.tokenizer.decode(labels[i], skip_special_tokens=True))
            pbar.update(1)
        
        result = self.compute_metrics((preds, refs))
        if not os.path.exists(self.args.output_dir + f"/{self.gettime()}_{self.modelname}/"):
            os.mkdir(self.args.output_dir + f"/{self.gettime()}_{self.modelname}/")
        log_name = self.args.output_dir + f"/{self.gettime()}_{self.modelname}/eval_{self.args.eval_strategy}_{num}.txt"
        self.save_txt([[cons[i], preds[i], refs[i]] for i in range(len(preds))], log_name, result)
        self.save_model(self.model, self.args.output_dir + f"/{self.gettime()}_{self.modelname}/{self.args.eval_strategy}_{num}.pt")
        write_data = [self.modelname, f"{self.args.eval_strategy}_{num}"]+ list(result.values())
        self.report_result(write_data)

    @torch.no_grad()
    def accelerate_valid_step(self, num):
        #     accelerator.wait_for_everyone()
        #     unwrap_p2r = accelerator.unwrap_model(p2r_model)
        #     valid_step(unwrap_p2r, valid_loader, valid_references, modelname, total_step, args)
        #     save_model(unwrap_p2r, save_path+'/'+modelname, total_step, 'step') 
        self.model.eval()
        pbar = tqdm(enumerate(self.test_loader),total=len(self.test_loader))
        refs = []
        preds = []
        cons = []
        for idx, data in pbar:
            input, context, labels = self.get_eval_input(data)
            output = self.accelerator.unwrap_model(self.model).generate(**input, max_length=40, num_beams=4, early_stopping=True)
            output = self.accelerator.pad_across_processes(
                    output, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            output = output.to('cpu').tolist()
            for i in range(len(output)):
                cons.append(self.tokenizer.decode(context[i], skip_special_tokens=True))
                preds.append(self.tokenizer.decode(output[i], skip_special_tokens=True))
                refs.append(self.tokenizer.decode(labels[i], skip_special_tokens=True))
            pbar.update(1)
        
        result = self.compute_metrics((preds, refs))
        if not os.path.exists(self.args.output_dir + f"/{self.gettime()}_{self.modelname}/"):
            os.mkdir(self.args.output_dir + f"/{self.gettime()}_{self.modelname}/")
        log_name = self.args.output_dir + f"/{self.gettime()}_{self.modelname}/eval_{self.args.eval_strategy}_{num}.txt"
        self.save_txt([[cons[i], preds[i], refs[i]] for i in range(len(preds))], log_name, result)
        self.save_model(self.model, self.args.output_dir + f"/{self.gettime()}_{self.modelname}/{self.args.eval_strategy}_{num}.pt")
        write_data = [self.modelname, f"{self.args.eval_strategy}_{num}"]+ list(result.values())
        self.report_result(write_data)   
    
    def compute_metrics(self, eval_pred):
        preds, refs = eval_pred
        if not isinstance(preds[0],str):
            preds = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
            refs = np.where(refs != -100, refs, self.tokenizer.pad_token_id)
            refs = [self.tokenizer.decode(ref, skip_special_tokens=True) for ref in refs]
        result = {}
        result["bleu"] = sacrebleu.compute(
                            predictions=preds,
                            references=refs,
                        )["score"]
        score_rouge = rouge.compute(
                            predictions=preds,
                            references=refs,
                            tokenizer=lambda x: self.tokenizer.tokenize(x)
                        )
        result["R1"] = score_rouge["rouge1"]*100
        result["R2"] = score_rouge["rouge2"]*100
        result["RL"] = score_rouge["rougeL"]*100
        try:
            score_bert = bertscore.compute(
                                predictions=preds, 
                                references=refs, 
                                lang="en",
                            )
        except:
            score_bert = bertscore.compute(
                                predictions=preds, 
                                references=refs, 
                                lang="en",
                                device="cpu"
                            )
        result["bertscore"] = np.mean(score_bert["f1"])*100

        return result
    
    def train(self):
        print("Preparing data...")
        self.prepare_loader()
        self.prepare_model()
        self.prepare_optimizer()
        self.total_step = 0

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        assert self.train_loader is not None
        assert self.test_loader is not None

        print("Start training...")
        self.model.to(self.device)
        for epoch in range(self.args.n_epoch):
            self.train_step(epoch)
            if self.args.eval_strategy == "epoch":
                self.valid_step(epoch)

    def preprocess_function(self, examples):
        inputs = examples["text"]
        model_inputs = self.tokenizer(inputs, max_length=32, truncation=True)
        labels = self.tokenizer(examples["labels"], max_length=32, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def quick_baseline(self):
        train_raw_dataset = HDataset.from_dict({"text": [x[0] for x in self.train_data], "labels": [x[1] for x in self.train_data]})
        test_raw_dataset = HDataset.from_dict({"text": [x[0] for x in self.test_data], "labels": [x[1] for x in self.test_data]})
        dataset = DatasetDict(
            {
                'train' : train_raw_dataset.map(self.preprocess_function, batched=True),
                'test' : test_raw_dataset.map(self.preprocess_function, batched=True)
            }
        )

        seq2seq_args = Seq2SeqTrainingArguments(
            "test-summarization",
            evaluation_strategy = self.args.eval_strategy,
            learning_rate=self.args.lr,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            weight_decay=0.01,
            save_total_limit=5,
            num_train_epochs=self.args.n_epoch,
            predict_with_generate=True,
        )

        trainer = Seq2SeqTrainer(
            self.model,
            args=seq2seq_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.evaluate()
        trainer.save_model(self.args.output_dir + f"/{self.gettime()}_{self.modelname}/")
    
    def accelerate_train(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs]) 
        self.accelerator.print("Preparing data...")
        self.prepare_loader()
        self.prepare_model()
        self.prepare_optimizer()
        self.total_step = 0
        self.ddp = 1

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        assert self.train_loader is not None
        assert self.test_loader is not None

        self.train_loader, self.test_loader = self.accelerator.prepare(self.train_loader, self.test_loader)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

        self.accelerator.print("Start training...")
        for epoch in range(self.args.n_epoch):
            self.accelerate_train_step(epoch)
            if self.args.eval_strategy == "epoch":
                self.accelerate_valid_step(epoch)

def main():
    trainer = TextTrainer()
    args = parse_args()
    trainer.modelname = getname(args)
    trainer.csv_path = "/share/zhangzk/Organized_Coding/output.csv"

    set_seed(args.seed)

    trainer.args = args
    trainer.device = f'cuda:{args.gpu}'

    trainer.model = BartForConditionalGeneration.from_pretrained('/share/zhangzk/Model/bart')
    trainer.tokenizer = BartTokenizer.from_pretrained('/share/zhangzk/Model/bart')

    train_dataset = trainer.load_txt('/share/zhangzk/Organized_Coding/demo/train.txt')
    test_dataset = trainer.load_txt('/share/zhangzk/Organized_Coding/demo/test.txt')
    
    trainer.train_data = train_dataset
    trainer.test_data = test_dataset

    t1 = time.time()
    if args.mode == "normal":
        trainer.train()
    elif args.mode == "baseline":
        trainer.quick_baseline()
    elif args.mode == "accelerate":
        accelerate_set_seed(args.seed)
        trainer.accelerate_train()
    else:
        raise NotImplementedError
    t2 = time.time()
    print(f"Time cost: {t2-t1}")

main()




