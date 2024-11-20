import pandas as pd

MODEL_NAME = "distilbert-base-uncased"
MODEL_OUT_DIR = "english-dataset.model"


def get_dataset():
    f = open("datasets/un/data.txt")

    sentences = [[]]
    labels = [[]]
    for line in f:
        if line[0] == "\n":
            sentences.append([])
            labels.append([])
            continue
        if line[0] == "#":
            continue

        spline = line.split("\t")

        sentences[-1].append(spline[1])
        labels[-1].append(spline[2])

    f.close()

    df = pd.DataFrame({"tokens": sentences, "ner_tags": labels})

    return df
