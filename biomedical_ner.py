import transformers as ts
from datasets import Dataset
from datasets import load_dataset

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import pandas as pd
import math
import csv

from minialbert_modeling import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasetName = "NCBI-disease" #Dataset name

pretrainedPath = "miniALBERT/models/bio-miniALBERT" #Path to the pre-trained model
tokenizerPath = "bert-base-cased"

datasetPath = f"biobert-datasets/datasets/NER/{datasetName}/" #Path to the preprocessed dataset (downloaded from https://github.com/dmis-lab/biobert)
logsPath = pretrainedPath + f"/ner_logs/{datasetName}-logs.txt"
modelPath = pretrainedPath + f"/final/model"

try:
  with open(logsPath, mode="w") as f:
    f.write("")
except:
  pass

"""#Dataset Utilities"""

def load_ner_dataset(folder):
  allLabels = set(pd.read_csv(folder + "train.tsv", sep="\t", header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')[1])

  label_to_index = {label: index for index , label in enumerate(allLabels)}
  index_to_label = {index: label for index , label in enumerate(allLabels)}

  def load_subset(subset):
    lines = []

    with open(folder + subset, mode="r") as f:
      lines = f.readlines()

    sentences = []
    labels = []

    currentSampleTokens = []
    currentSampleLabels = []

    for line in lines:
      if line.strip() == "":
        sentences.append(currentSampleTokens)
        labels.append(currentSampleLabels)
        currentSampleTokens = []
        currentSampleLabels = []
      else:
        cleanedLine = line.replace("\n","")
        token , label = cleanedLine.split("\t")[0].strip() , cleanedLine.split("\t")[1].strip()
        currentSampleTokens.append(token)
        currentSampleLabels.append(label_to_index[label])
    
    dataDict = {
        "tokens": sentences,
        "ner_tags": labels,
    }
    
    return Dataset.from_dict(dataDict)
  
  trainingDataset = load_subset("train.tsv")
  validationDataset = Dataset.from_dict(load_subset("train_dev.tsv")[len(trainingDataset):])
  testDataset = load_subset("test.tsv")

  return {
      "train": trainingDataset,
      "validation": validationDataset,
      "test": testDataset,
      "all_ner_tags": list(allLabels),
  }

"""#Loading Dataset"""

dataset = load_ner_dataset(datasetPath)

print(dataset)

label_names = dataset["all_ner_tags"]

tokenizer = ts.AutoTokenizer.from_pretrained(tokenizerPath, use_auth_token=True)

#Get the values for input_ids, token_type_ids, attention_mask
def tokenize_adjust_labels(all_samples_per_split, **kargs):
  tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True, max_length=512, padding=kargs["padding"])  
  total_adjusted_labels = []

  for k in range(0, len(tokenized_samples["input_ids"])):
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids(batch_index=k)
    existing_label_ids = all_samples_per_split["ner_tags"][k]
    i = -1
    adjusted_label_ids = []

    for wid in word_ids_list:
      if(wid is None):
        adjusted_label_ids.append(-100)
      elif(wid!=prev_wid):
        i = i + 1
        adjusted_label_ids.append(existing_label_ids[i])
        prev_wid = wid
      else:
        adjusted_label_ids.append(existing_label_ids[i])
        
    total_adjusted_labels.append(adjusted_label_ids)

  tokenized_samples["labels"] = total_adjusted_labels
  
  return tokenized_samples

tokenizedTrainDataset = dataset["train"].map(tokenize_adjust_labels, batched=True, remove_columns=dataset["train"].column_names, fn_kwargs={"padding": "do_not_pad"})
tokenizedValDataset = dataset["validation"].map(tokenize_adjust_labels, batched=True, remove_columns=dataset["validation"].column_names, fn_kwargs={"padding": "max_length"})
tokenizedTestDataset = dataset["test"].map(tokenize_adjust_labels, batched=True, remove_columns=dataset["test"].column_names, fn_kwargs={"padding": "max_length"})
"""#Model

#Training
"""

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }

    return flattened_results

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

def evaluate(model):
  model.eval()

  predictions = []
  labels = []

  for index, item in enumerate(tokenizedTestDataset):
    sample = data_collator([item])
    for key, value in sample.items():
      sample[key] = value.cuda()
    output = model(**sample).logits.cpu().detach().numpy()

    if len(predictions) == 0:
      predictions = output
      labels = sample["labels"].cpu().numpy()
    else:
      predictions = np.concatenate((predictions, output), axis=0)
      labels = np.concatenate((labels, sample["labels"].cpu().numpy()), axis=0)

    if index % 100 == 0:
      print(index)
    
  predictions = np.array(predictions)
  labels = np.array(labels)

  return compute_metrics((predictions, labels))

def trainAndEvaluate(lr, batchsize):
  model = MiniAlbertForTokenClassification.from_pretrained(modelPath, num_labels=len(label_names))

  trainingArguments = ts.TrainingArguments(
      "output/",
      seed=42,
      logging_steps=250,
      save_steps= 2500,
      num_train_epochs=5,
      learning_rate=lr,
      lr_scheduler_type="cosine",
      per_device_train_batch_size=batchsize,
      per_device_eval_batch_size=1,
      weight_decay=0.01,
  )

  trainer = ts.Trainer(
      model=model,
      args=trainingArguments,
      train_dataset=tokenizedTrainDataset,
      eval_dataset=tokenizedValDataset,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
  )

  trainer.train()

  testResult = evaluate(model)

  with open(logsPath, mode="a+") as f:
    f.write(f"---HyperParams---\nBatchsize= {batchsize} Lr= {lr}\n---Test Results---\n{str(testResult)}\n\n")

learningRates = [5e-5]
batchsizes = [16]

for lr in learningRates:
  for batchsize in batchsizes:
    for _ in range(3):     
      trainAndEvaluate(lr=lr, batchsize=batchsize)
