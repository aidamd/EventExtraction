import pandas as pd
import os.path
import math
from utils import *
from nltk import sent_tokenize
import nltk.tokenize as tokenizer
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def GenerateTrain(count):
    if os.path.isfile("train_anno.csv"):
        df = pd.read_csv("train_anno.csv")
        # should I remove stop words?

    elif os.path.isfile("patch_sample_annotated.csv"):
        propublica = pd.read_csv("propublica.csv")["text"].dropna()

        print("Number of articles in Propublica:", propublica.shape[0])
        print("Getting", count / 2, "random articles from Propublica")
        propublica_sample = propublica.sample(1000).tolist()
        propublica_sample = [re.sub(r"[\s]+", " ", sent) for sent in propublica_sample]
        labels = [1 for i in range(len(propublica_sample))]

        propublica_df = pd.DataFrame.from_dict({"text": propublica_sample, "labels": labels})
        patch_sample = pd.read_csv("patch_sample_annotated.csv")

        df = propublica_df.append(patch_sample).sample(count)
        print("Train set shape:", df.shape)
        df.to_csv("train.csv", index=False)
    else:
        patch = pd.read_csv("patch.csv")["text"].dropna()
        print("Number of articles in Patch:", patch.shape[0])
        print("Getting", count / 2, "random articles from each set")
        patch_sample = patch.sample(math.floor(count / 2)).tolist()

        df = pd.DataFrame.from_dict({"text": patch_sample, "labels": [0 for i in range(len(patch_sample))]})

        print("Output the patch sample to be annotated")
        df.to_csv("patch_sample.csv", index=False)
        print("Please annotate the patch sample first and then save it as 'patch_sample_annotated.csv'")
        exit(1)

    vocabs = get_vocabs()
    bags = TrainToBags(df, vocabs)

    train, dev = train_test_split(bags, test_size=0.2, random_state=33)

    patch = pd.read_csv("patch.csv")
    print("Test set shape:", patch.shape)

    """
    print("Converting test dataset to batches")
    test = TrainToBags(patch, vocabs, True)
    """
    pickle.dump((train, dev, vocabs), open("data.pkl", "wb"))
    return (train, dev, vocabs)

def TrainToBags(df, vocab, test=False):
    dictionary = {word: idx for idx, word in enumerate(vocab)}
    batches = list()
    print("Cleaning data ...")
    with tqdm(total=df.shape[0]) as counter:
        for idx, row in df.iterrows():
            try:
                text = clean(row["text"])
            except Exception:
                continue
            words = [tokenizer.TreebankWordTokenizer().tokenize(sent) for sent in sent_tokenize(text)]
            bag, max_len, lengths = bag_to_ids(dictionary, words)
            if test:
                batches.append((bag, lengths))
            else:
                batches.append((bag, lengths, row["labels"], row["target"], row["action"]))
            counter.update(1)
    return batches

