import torch
import torch.nn as nn
from torch.functional import F

import transformers as ts
import datasets as ds
from datasets import Dataset, load_dataset, load_from_disk

import numpy as np
import random

random.seed(42)

ds = load_dataset("wikipedia", "20220301.en", cache_dir="dataset/wikipedia")
tokenizer = ts.AutoTokenizer.from_pretrained("bert-base-uncased")

k = 10

def mappingFunction(dataset):
  outputs = {
      "input_ids": [],
      "attention_mask": [],
      "token_type_ids": [],
      "special_tokens_mask": [],
  }

  for text in dataset["text"]:
    output = tokenizer(
               text,
               max_length=256,
               truncation=True,
               stride=128,
               return_overflowing_tokens=True,
               return_special_tokens_mask=True,
    )

    if len(output["input_ids"]) <= k:
      input_ids, attention_mask, token_type_ids, token_mask = (output["input_ids"], output["attention_mask"], output["token_type_ids"], output["special_tokens_mask"])
    else:
      input_ids, attention_mask, token_type_ids, token_mask = zip(*random.sample(list(zip(output["input_ids"], output["attention_mask"], output["token_type_ids"], output["special_tokens_mask"])), k))

    outputs["input_ids"] += input_ids
    outputs["attention_mask"] += attention_mask
    outputs["token_type_ids"] += token_type_ids
    outputs["special_tokens_mask"] += token_mask

  return outputs

dataset = ds["train"].map(mappingFunction, remove_columns=ds["train"].column_names, batched=True)

datasetPath = "tokenizedDatasets/wikipedia-256/"

dataset.save_to_disk(datasetPath)

print(load_from_disk(datasetPath))

