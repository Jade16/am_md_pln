#!/usr/bin/env python3
import streamlit as st
import wandb

import un
import data_exploration
import trainer
import config

wandb.init(mode="disabled")

dataset = None
model_name = None
model_dir = None

if config.MODEL == "un":
    dataset = un.get_dataset()
    model_name = un.MODEL_NAME
    model_dir = un.MODEL_OUT_DIR
else:
    raise ValueError("Invalid model selected, use --help to se options")

st.title("Reconhecimento e Desambiguação de Entidade Nomeada")

data_exploration.streamlit_show(dataset)

tokenizer, model = None, None

if config.DO_TRAINING:
    trainer.train_model(dataset, model_name, model_dir)

tokenizer, model = trainer.get_pretrained_model(model_dir)

data = """
Before proceeding further, I should like to inform members that action on draft resolution iv, entitled situation of human rights of Rohingya Muslims and other minorities in Myanmar is postponed to a later date to allow time for the review of its programme budget implications by the fifth committee. The assembly will take action on draft resolution iv as soon as the report of the fifth committee on the programme budget implications is available. I now give the floor to delegations wishing to deliver explanations of vote or position before voting or adoption.
"""

words, tags = trainer.predict(tokenizer, model, data)
print(words, tags)
