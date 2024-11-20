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
model_out_dir = None

if config.MODEL == "un":
    dataset = un.get_dataset()
    model_name = un.MODEL_NAME
    model_out_dir = un.MODEL_OUT_DIR
else:
    raise ValueError("Invalid model selected, use --help to se options")

st.title("Reconhecimento e Desambiguação de Entidade Nomeada")

data_exploration.streamlit_show(dataset)

tokenizer, model = None, None

if config.DO_TRAINING:
    trainer.train_model(dataset, model_name, model_out_dir)

tokenizer, model = trainer.get_pretrained_model()
