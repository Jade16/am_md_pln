from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

import config

label_list = [
    "O",
    "B-MISC",
    "I-MISC",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]

label_encoding_dict = {
    "I-PRG": 2,
    "I-I-MISC": 2,
    "I-OR": 6,
    "O": 0,
    "I-": 0,
    "VMISC": 0,
    "B-PER": 3,
    "I-PER": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-LOC": 7,
    "I-LOC": 8,
    "B-MISC": 1,
    "I-MISC": 2,
}


def tokenize_and_align_labels(examples, base_model_name):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenized_inputs = tokenizer(
        list(examples["tokens"]), truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == "0":
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_model(df, base_model_name, model_output_dir):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    train, test = Dataset.from_pandas(df.iloc[:800]), Dataset.from_pandas(df.iloc[800:])

    tokenized_train = train.map(
        lambda e: tokenize_and_align_labels(e, base_model_name), batched=True
    )
    tokenized_test = test.map(
        lambda e: tokenize_and_align_labels(e, base_model_name), batched=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name, num_labels=len(label_list)
    )

    args = TrainingArguments(
        "test-ner",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
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
    trainer.save_model(model_output_dir)

def get_pretrained_model():
    return (None, None)


def predict(model):
    return None
