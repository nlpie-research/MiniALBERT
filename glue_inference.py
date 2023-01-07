import transformers as ts
from datasets import Dataset
from datasets import load_dataset

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
from transformers import DataCollatorForTokenClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from minialbert_modeling import *

import numpy as np
import pandas as pd
import math
import csv

class DatasetInfo:
  def __init__(self, name, isMultiSentence=False, validationSubsets=["validation"], lr=[5e-5, 5e-4, 1e-3], batch_size=[32], epochs=3, runs=1):
    self.name = name
    self.isMultiSentence = isMultiSentence
    self.validationSubsets = validationSubsets
    self.lr = lr
    self.batch_size = batch_size
    self.epochs = epochs
    self.runs = runs

class ModelInfo:
  def __init__(self, pretrainedPath, modelPath, isCustom=True, isAdapterTuning=False, use_token_type_ids=True):
    self.pretrainedPath = pretrainedPath
    self.modelPath = modelPath

    self.logsPath = pretrainedPath + f"/glue/"

    self.isCustom = isCustom
    self.isAdapterTuning = isAdapterTuning
    self.use_token_type_ids = use_token_type_ids

  def get_logs_path(self, datasetName):
    return self.logsPath + f"{datasetName}-logs.txt" if not self.isAdapterTuning else self.logsPath + f"{datasetName}-adapter-logs.txt"
  
  def load_model(self, num_labels):
    if self.isCustom:
      model = MiniAlbertForSequenceClassification.from_pretrained(self.modelPath, num_labels=num_labels)
      
      if self.isAdapterTuning:
        model.trainAdaptersOnly()
    else:
      model = ts.AutoModelForSequenceClassification.from_pretrained(self.modelPath, num_labels=num_labels)
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = [
    DatasetInfo("stsb", True, epochs=5, batch_size=[16], runs=3),
    DatasetInfo("cola", False, epochs=5, batch_size=[16], runs=3),
    DatasetInfo("mrpc", True, epochs=5, batch_size=[16], runs=3),
    DatasetInfo("rte", True, epochs=5, batch_size=[16], runs=3),
    DatasetInfo("qqp", True),
    DatasetInfo("qnli", True),
    DatasetInfo("mnli", True, ["validation_matched", "validation_mismatched"]),
    DatasetInfo("sst2", False),
]

models = [
    ModelInfo("miniALBERT/models/general-miniALBERT-adatper",
              "miniALBERT/models/general-miniALBERT-adatper/final/model",
              isCustom=True,
              isAdapterTuning=False,),
    ModelInfo("miniALBERT/models/general-miniALBERT",
              "miniALBERT/models/general-miniALBERT/final/model",
              isCustom=True,
              isAdapterTuning=False,),
    ModelInfo("miniALBERT/models/general-miniALBERT-adatper-ef",
              "miniALBERT/models/general-miniALBERT-adatper-ef/final/model",
              isCustom=True,
              isAdapterTuning=False,),
    ModelInfo("miniALBERT/models/general-miniALBERT-ef",
              "miniALBERT/models/general-miniALBERT-ef/final/model",
              isCustom=True,
              isAdapterTuning=False,),
    ModelInfo("miniALBERT/models/general-miniALBERT-adatper",
              "miniALBERT/models/general-miniALBERT-adatper/final/model",
              isCustom=True,
              isAdapterTuning=True,),
    ModelInfo("miniALBERT/models/general-miniALBERT",
              "miniALBERT/models/general-miniALBERT/final/model",
              isCustom=True,
              isAdapterTuning=True,),
    ModelInfo("miniALBERT/models/general-miniALBERT-adatper-ef",
              "miniALBERT/models/general-miniALBERT-adatper-ef/final/model",
              isCustom=True,
              isAdapterTuning=True,),
    ModelInfo("miniALBERT/models/general-miniALBERT-ef",
              "miniALBERT/models/general-miniALBERT-ef/final/model",
              isCustom=True,
              isAdapterTuning=True,),
]

tokenizerPath = "bert-base-uncased"
tokenizer = ts.AutoTokenizer.from_pretrained(tokenizerPath)
data_collator = ts.DataCollatorWithPadding(tokenizer, return_tensors="pt")

def load_datasets(info):
  """#Dataset Utilities"""

  dataset = load_dataset("glue", info.name)

  if info.name != "stsb":
    num_labels = len(set(dataset["train"]["label"]))
  else:
    num_labels = 1

  def mappingFunction(samples, **kargs):
    if info.isMultiSentence:
      outputs = tokenizer(samples[dataset["train"].column_names[0]],
                          samples[dataset["train"].column_names[1]],
                          truncation=True,
                          padding=kargs["padding"])
    else:
      outputs = tokenizer(samples["sentence"],
                          truncation=True,
                          padding=kargs["padding"])

    outputs["labels"] = samples["label"]

    return outputs

  tokenizedTrainDataset = dataset["train"].map(mappingFunction,
                                              batched=True,
                                              remove_columns=dataset["train"].column_names,
                                              fn_kwargs={"padding": "do_not_pad"})

  tokenizedValDatasets = []

  for name in info.validationSubsets:
    tokenizedValDataset = dataset[name].map(mappingFunction,
                                            batched=True,
                                            remove_columns=dataset[name].column_names,
                                            fn_kwargs={"padding": "max_length"})
    
    tokenizedValDatasets.append(tokenizedValDataset)

  return tokenizedTrainDataset, tokenizedValDatasets, num_labels

def evaluate(model, info, valDataset):
  model.eval()

  metric = load_metric("glue", info.name)

  for index, row in enumerate(valDataset):
      sample = data_collator([row])

      for key, value in sample.items():
        sample[key] = value.cuda()

      if info.name != "stsb":
        output = np.argmax(model(**sample).logits.cpu().detach().numpy(), axis=-1)
      else:
        output = model(**sample).logits.cpu().detach().numpy()

      metric.add_batch(predictions=output, references=sample["labels"].cpu().detach().numpy())

      if index % 100 == 0:
        print(index)

  return metric.compute()

def initLogsFile(path):
  try:
    with open(path, mode="w") as f:
      f.write("")
  except:
    pass

def trainAndEvaluate(modelInfo, dsInfo, trainDataset, valDatasets, num_labels):
  logsPath = modelInfo.get_logs_path(dsInfo.name)
  initLogsFile(logsPath)

  if not modelInfo.use_token_type_ids:
    trainDataset = trainDataset.remove_columns(["token_type_ids"])

  for lr in dsInfo.lr:
    for batch_size in dsInfo.batch_size:
      for _ in range(dsInfo.runs):
        model = modelInfo.load_model(num_labels)

        trainingArguments = ts.TrainingArguments(
            "output/",
            seed=42,
            logging_steps=250,
            save_steps= 2500,
            num_train_epochs=dsInfo.epochs,
            learning_rate=lr,
            lr_scheduler_type="linear",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
        )

        trainer = ts.Trainer(
            model=model,
            args=trainingArguments,
            train_dataset=trainDataset,
            data_collator=data_collator,
        )

        trainer.train()

        for name, dataset in zip(dsInfo.validationSubsets, valDatasets):
          if not modelInfo.use_token_type_ids:
            dataset = dataset.remove_columns(["token_type_ids"])
          result = evaluate(model, dsInfo, dataset)
          with open(logsPath, mode="a+") as f:
            f.write(f"---HyperParams---\nBatchsize= {batch_size} Lr= {lr}\n---{name} results---\n{str(result)}\n\n")

for dsInfo in datasets:
  trainDataset, valDatasets, num_labels = load_datasets(dsInfo)
  for modelInfo in models:
    trainAndEvaluate(modelInfo, dsInfo, trainDataset, valDatasets, num_labels)
