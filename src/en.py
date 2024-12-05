import re
import pandas as pd
import nltk


MODEL_NAME = "distilbert-base-uncased"
MODEL_OUT_DIR = "english-dataset.model"

symbols = {"@": "LOC", "&": "PER", "§": "MISC"}

def phrase_to_token_tag_list(phrase):
    tokenized = iter(nltk.word_tokenize(phrase))

    tokens = []
    ner_tags = []

    for tok in tokenized:
        if tok not in symbols.keys():
            tokens.append(tok)
            ner_tags.append("O")
            continue

        try:
            if tokenized.__next__() != "(":
                print("invalid continuation, ignoring entity")
                continue

            first_entity_token = tokenized.__next__()
        except StopIteration:
            print("sentence ends at the middle of an entity")
            continue

        if first_entity_token == ")":
            print("empty entity")
            continue

        tokens.append(first_entity_token)
        ner_tags.append("B-" + symbols[tok])

        for next_entity_tokens in tokenized:
            if next_entity_tokens == ")":
                break
            tokens.append(next_entity_tokens)
            ner_tags.append("I-" + symbols[tok])

    return tokens, ner_tags



def get_dataset():
    dataset = {"tokens": [], "ner_tags": []}

    for i in range(1,6):
        f = open(f"datasets/en/{i}.txt")
        data = f.read()
        f.close()

        in_bracket = False

        in_parenthesis = False
        out = [""]

        for c in data:
            if in_bracket and c == "]":
                in_bracket = False
                continue
            if in_bracket:
                continue
            if c == "[":
                in_bracket = True
                continue

            if c == "(":
                in_parenthesis = True
            if c == ")":
                in_parenthesis = False

            if not in_parenthesis and c in ".\n":
                out.append("")
            else:
                out[-1] += c

        for phrase in out:
            if re.match(r"Letter|MB: |[0-9].*page", phrase):
                continue
            if not re.match(r".*[a-zA-Z]", phrase):
                continue

            tokens, ner_tags = phrase_to_token_tag_list(phrase)

            dataset["tokens"].append(tokens)
            dataset["ner_tags"].append(ner_tags)

    df = pd.DataFrame(dataset)
    return df
