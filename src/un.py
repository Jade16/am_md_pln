import pandas as pd

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


def train_model(df):
    return (None, None)


def get_pretrained_model():
    return (None, None)


def predict(model):
    return None
