#!/usr/bin/env python3
import io

import streamlit as st
import wandb
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

import config
import data_exploration
import trainer
import un
import en
import pt

wandb.init(mode="disabled")

dataset = None
model_name = None
model_dir = None

if config.MODEL == "un":
    dataset = un.get_dataset()
    model_name = un.MODEL_NAME
    model_dir = un.MODEL_OUT_DIR
elif config.MODEL == "en":
    dataset = en.get_dataset()
    model_name = en.MODEL_NAME
    model_dir = en.MODEL_OUT_DIR
elif config.MODEL == "pt":
    dataset = pt.get_dataset()
    model_name = pt.MODEL_NAME
    model_dir = pt.MODEL_OUT_DIR
else:
    raise ValueError("Invalid model selected, use --help to see options")

st.set_page_config(layout="wide")
st.title("Reconhecimento e Desambiguação de Entidade Nomeada")

st.header("Exploração dos Dados")
data_exploration.streamlit_show(dataset)

train, test = train_test_split(dataset, test_size=config.TEST_SIZE)
train = Dataset.from_pandas(pd.DataFrame(train, columns=["tokens", "ner_tags"]))
test = Dataset.from_pandas(pd.DataFrame(test, columns=["tokens", "ner_tags"]))

tokenizer, model = None, None

if config.DO_TRAINING:
    trainer.train_model(train, test, model_name, model_dir)

tokenizer, model = trainer.get_pretrained_model(model_dir)

st.header("Avaliação do Modelo")
trainer.evaluate(tokenizer, model, test)

st.header("Predição Interativa")
uploaded_file = st.file_uploader("Escolha um Arquivo para Realizar NER")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    data = stringio.read()

    result = trainer.predict(tokenizer, model, data)
    st.table(result)
