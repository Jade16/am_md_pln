#!/usr/bin/env python3
import io

import streamlit as st
import wandb

import config
import data_exploration
import trainer
import un

wandb.init(mode="disabled")

dataset = None
model_name = None
model_dir = None

if config.MODEL == "un":
    dataset = un.get_dataset()
    model_name = un.MODEL_NAME
    model_dir = un.MODEL_OUT_DIR
else:
    raise ValueError("Invalid model selected, use --help to see options")

st.title("Reconhecimento e Desambiguação de Entidade Nomeada")

data_exploration.streamlit_show(dataset)

tokenizer, model = None, None

if config.DO_TRAINING:
    trainer.train_model(dataset, model_name, model_dir)

tokenizer, model = trainer.get_pretrained_model(model_dir)

uploaded_file = st.file_uploader("Escolha um Arquivo para Realizar NER")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    data = stringio.read()

    result = trainer.predict(tokenizer, model, data)
    st.table(result)
