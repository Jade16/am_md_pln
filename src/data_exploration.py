import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import wordcloud


def streamlit_show(df):
    fig, ax = plt.subplots()
    df["tokens"].map(lambda ts: len(ts)).plot(kind="hist", ax=ax)
    st.pyplot(fig)

    countings = (
        df.explode("tokens")
        .groupby("tokens")
        .count()
        .sort_values("ner_tags", ascending=False)
    )
    puncts = countings[
        np.logical_not(countings.index.str.contains("[a-zA-Z0-9]", regex=True))
    ]
    fig, ax = plt.subplots()
    puncts.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    words = countings[countings.index.str.contains("[a-zA-Z0-9]", regex=True)]
    words[:30].plot(kind="bar", ax=ax)
    st.pyplot(fig)

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
    fig, ax = plt.subplots()
    entity_counts[:30].plot(kind="bar", ax=ax)
    st.pyplot(fig)

    ner_counts = (
        df.explode("ner_tags")
        .groupby("ner_tags")
        .count()
        .sort_values("tokens", ascending=False)
    )
    fig, ax = plt.subplots()
    ner_counts[1:].plot(kind="bar", ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    entity_counts.groupby("tag").count().sort_values("count", ascending=False).plot(
        kind="bar", ax=ax
    )
    st.pyplot(fig)

    for kind in ner_counts.index[1:]:
        fig, ax = plt.subplots()
        df.map(lambda ts: sum([1 if i == kind else 0 for i in ts])).groupby(
            "ner_tags"
        ).count().sort_values("tokens", ascending=False)[1:].plot(kind="bar", ax=ax)
        ax.set_title(kind)
        st.pyplot(fig)

    w = wordcloud.WordCloud(background_color="white").generate(
        " ".join([" ".join(i) for i in df["tokens"]])
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(w, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
