#!/usr/bin/env python3
import argparse

import streamlit as st
import wandb

import data_exploration
import un

wandb.init(mode="disabled")

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

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    type=str,
    default="un",
    help="model and dataset to use, possible options are un, en, pt",
)

parser.add_argument(
    "--do-training",
    type=bool,
    default=False,
    help="Train the model, if not used just uses the pretrained one or fails",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size used during training (ignored if not training)"
)

args = parser.parse_args()

provider = None

if args.model == "un":
    provider = un
else:
    raise ValueError("Invalid model selected, use --help to se options")

dataset = provider.get_dataset()

tokenizer, model = None, None

if args.do_training:
    tokenizer, model = provider.train_model(dataset)
else:
    tokenizer, model = provider.get_pretrained_model()


st.title("Reconhecimento e Desambiguação de Entidade Nomeada")

data_exploration.streamlit_show(dataset)


