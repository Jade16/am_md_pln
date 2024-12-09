import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import wordcloud
import altair as alt


def streamlit_show(df):
    st.header("Histograma do Número de Tokens por Sentença")
    st.bar_chart(df["tokens"].map(lambda ts: len(ts)).value_counts())

    countings = (
        df.explode("tokens")
        .groupby("tokens")
        .count()
        .sort_values("ner_tags", ascending=False)
    )
    puncts = countings[
        np.logical_not(countings.index.str.contains("[a-zA-Z0-9]", regex=True))
    ]

    c = (
        alt.Chart(puncts.reset_index())
        .mark_bar()
        .encode(x=alt.X("tokens", sort=None), y="ner_tags")
    )
    st.header("Número de Ocorrências de pontuações")
    st.altair_chart(c, use_container_width=True)

    words = countings[
        countings.index.str.contains("[a-zA-Z0-9]", regex=True)
    ].sort_values(by="ner_tags", ascending=False)

    c = (
        alt.Chart(words[:30].reset_index())
        .mark_bar()
        .encode(x=alt.X("tokens", sort=None), y="ner_tags")
    )
    st.header("Número de Ocorrências das 30 palavras mais comuns")
    st.altair_chart(c, use_container_width=True)

    arr = []
    for i, j in zip(df["tokens"], df["ner_tags"]):
        for k, l in zip(i, j):
            arr.append([k, l])

    exploded_df = pd.DataFrame(arr, columns=["token", "tag"])
    exploded_df["count"] = 1
    entity_counts = (
        exploded_df[exploded_df["tag"] != "O"]
        .groupby(["token", "tag"])
        .count()
        .sort_values("count", ascending=False)
    )

    entity_counts = entity_counts.reset_index()

    entity_counts["token_and_tag"] = entity_counts["token"] + " " + entity_counts["tag"]

    c = (
        alt.Chart(entity_counts[:30])
        .mark_bar()
        .encode(x=alt.X("token_and_tag", sort=None), y="count")
    )
    st.header("Número de Ocorrências dos 30 tokens compositores de entidades mais comuns")
    st.altair_chart(c, use_container_width=True)

    ner_counts = (
        df.explode("ner_tags")
        .groupby("ner_tags")
        .count()
        .sort_values("tokens", ascending=False)
    )
    c = (
        alt.Chart(ner_counts[1:].reset_index())
        .mark_bar()
        .encode(x=alt.X("ner_tags", sort=None), y="tokens")
    )
    st.header("Contagem da Quantidade de Tokens para Cada Tipo de Entidade")
    st.altair_chart(c, use_container_width=True)

    unique_entities = (
        entity_counts.groupby("tag").count().sort_values("count", ascending=False)
    )
    c = (
        alt.Chart(unique_entities.reset_index())
        .mark_bar()
        .encode(x=alt.X("tag", sort=None), y="count")
    )

    st.header("Contagem da Quantidade de Tokens Únicos para Cada Tipo de Entidade")
    st.altair_chart(c, use_container_width=True)

    occurrences = pd.DataFrame()

    for kind in ner_counts.index[1:]:
        occurrences_inner = (
            df.map(lambda ts: sum([1 if i == kind else 0 for i in ts]))
            .groupby("ner_tags")
            .count()
            .sort_values("tokens", ascending=False)[1:]
        )

        occurrences_inner["kind"] = kind

        occurrences_inner = occurrences_inner.reset_index()

        occurrences = pd.concat([occurrences, occurrences_inner])

    st.header("Quantidade de Sentenças com um Determinado Número de Tags")
    st.bar_chart(occurrences, x="ner_tags", y="tokens", color="kind")

    w = wordcloud.WordCloud(background_color="black").generate(
        " ".join([" ".join(i) for i in df["tokens"]])
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(w, interpolation="bilinear")
    ax.axis("off")
    st.header("Word Cloud do Dataset")
    st.pyplot(fig)
