import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import wandb
import wordcloud
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

wandb.init(mode="disabled")

label_list = ['O','B-MISC','I-MISC','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']

label_encoding_dict = {'I-PRG': 2,'I-I-MISC': 2, 'I-OR': 6, 'O': 0, 'I-': 0, 'VMISC': 0, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8, 'B-MISC': 1, 'I-MISC': 2}

task = "ner"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

f = open('datasets/un/data.txt')

sentences = [[]]
labels = [[]]
for line in f:
  if line[0] == "\n":
    sentences.append([])
    labels.append([])
    continue
  if line[0] == "#":
    continue

  spline = line.split('\t')

  sentences[-1].append(spline[1])
  labels[-1].append(spline[2])

f.close()

df = pd.DataFrame({'tokens': sentences, 'ner_tags': labels})

train, test = Dataset.from_pandas(df.iloc[:800]), Dataset.from_pandas(df.iloc[800:])


df["tokens"].map(lambda ts: len(ts)).plot(kind="hist")

countings = df.explode("tokens").groupby("tokens").count().sort_values("ner_tags", ascending=False)

puncts = countings[np.logical_not(countings.index.str.contains("[a-zA-Z0-9]", regex=True))]
puncts.plot(kind="bar")

words = countings[countings.index.str.contains("[a-zA-Z0-9]", regex=True)]
words[:30].plot(kind="bar")

arr = []
for i, j in zip(sentences, labels):
  for k, l in zip(i, j):
    arr.append([k, l])

exploded_df = pd.DataFrame(arr, columns=["token", "tag"])
exploded_df["count"] = 1
entity_counts = exploded_df[exploded_df["tag"] != "O"].groupby(["token", "tag"]).count().sort_values("count", ascending=False)
entity_counts[:30].plot(kind="bar")

ner_counts = df.explode("ner_tags").groupby("ner_tags").count().sort_values("tokens", ascending=False)
ner_counts[1:].plot(kind="bar")

entity_counts.groupby("tag").count().sort_values("count", ascending=False).plot(kind="bar")

print(exploded_df.groupby("count").count())

for kind in ner_counts.index[1:]:
  df.map(lambda ts: sum([1 if i == kind else 0 for i in ts])).groupby("ner_tags").count().sort_values("tokens", ascending=False)[1:].plot(kind="bar")
  plt.title(kind)
  plt.show()


w = wordcloud.WordCloud(background_color="white").generate(" ".join([" ".join(i) for i in sentences]))
plt.figure(figsize=(10,10))
plt.imshow(w, interpolation="bilinear")
plt.axis("off")
plt.show()

def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_train = train.map(tokenize_and_align_labels, batched=True)
tokenized_test = test.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

args = TrainingArguments(
    f"test-{task}",
    evaluation_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=1e-5,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.evaluate()
trainer.save_model('english-dataset.model')


tokenizer = AutoTokenizer.from_pretrained('english-dataset.model')

paragraph = '''
Before proceeding further, I should like to inform members that action on draft resolution iv, entitled situation of human rights of Rohingya Muslims and other minorities in Myanmar is postponed to a later date to allow time for the review of its programme budget implications by the fifth committee. The assembly will take action on draft resolution iv as soon as the report of the fifth committee on the programme budget implications is available. I now give the floor to delegations wishing to deliver explanations of vote or position before voting or adoption.
'''

tokens = tokenizer(paragraph)
torch.tensor(tokens['input_ids']).unsqueeze(0).size()

model = AutoModelForTokenClassification.from_pretrained('english-dataset.model', num_labels=len(label_list))
predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
predictions = torch.argmax(predictions.logits.squeeze(), axis=1)

result = [label_list[i] for i in predictions]

words = tokenizer.batch_decode(tokens['input_ids'])
df = pd.DataFrame({'ner': result, 'words': words})
for i, j in zip(df["words"], df["ner"]):
  print(i, j)


st.title("Reconhecimento e Desambiguação de Entidades")


