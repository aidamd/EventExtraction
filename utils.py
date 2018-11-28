from collections import Counter
import operator
import numpy as np
from nltk.corpus import wordnet, stopwords
import re

def get_vocabs():
    # Gets the list of words and characters in the dataset
    v = open("google-10000-english-usa.txt", "r").readlines()
    all = [r.replace("\n", "") for r in v]
    vocab = all + ["<unk>", "<pad>"]
    return vocab

def bag_to_ids(dic, bag):
    i_bag = list()
    max_len = max(len(sent) for sent in bag)
    lengths = list()
    for sent in bag:
        i_sent = list()
        for word in sent:
            try:
                i_sent.append(dic[word.lower()])
            except Exception:
                i_sent.append(dic["<unk>"])
        lengths.append(len(i_sent))
        while len(i_sent) < max_len:
            i_sent.append(dic["<pad>"])
        i_bag.append(np.array(i_sent))
    return np.array(i_bag), max_len, lengths

def stop_words(sent):
    stop_words = set(stopwords.words('english'))
    stop_words_exp = re.compile(r"({})\s+".format('|'.join(stop_words)))
    try:
        new_sent = stop_words_exp.sub(' ', sent)
    except TypeError:
        print(sent)
        new_sent = []
    return new_sent


def clean(sent):
    sent = stop_words(sent)
    sent = re.sub(r"[\s]+", " ", sent)
    return sent