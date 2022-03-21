import sys
sys.path.append("../")
import os
from utils.constant import *
from utils.help_func import clean_data_for_look
from sklearn.utils import shuffle
from utils.help_func import save_pickle
import gensim
import numpy as np
import pandas as pd
import random

def make_wv_matrix(data, word_vectors):
    wv_matrix = []
    for i in range(len(data["vocab"])):
        word = data["idx_to_word"][i]
        if word in word_vectors.vocab:
            wv_matrix.append(word_vectors.word_vec(word))
        else:
            wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    # one for UNK and one for zero padding
    wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
    wv_matrix.append(np.zeros(300).astype("float32"))
    wv_matrix = np.array(wv_matrix)
    return wv_matrix


def set_data(x, y, test_idx):
    data = {}
    data["train_x"], data["train_y"] = x[:test_idx], y[:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}
    return data


def divide_mr(word_vectors):
    x, y = [], []
    save_path = "../data/mr/split_mr.pkl"
    save_wv_matrix_path = "../data/mr/split_mr_wv_matrix.pkl"
    with open("../data/mr/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            line = clean_data_for_look(line)
            x.append(line.split())
            y.append(1)

    with open("../data/mr/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            line = clean_data_for_look(line)
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    test_idx = len(x) // 10 * 8

    data = set_data(x, y, test_idx)
    wv_matrix = make_wv_matrix(data, word_vectors)

    save_pickle(save_path, data)
    save_pickle(save_wv_matrix_path, wv_matrix)


def divide_na(word_vectors):
    #  -- b : business -- e : entertainment -- m : health -- t : science and technology
    def category2lable(category):
        if category == "b":
            return 0
        elif category == "e":
            return 1
        elif category == "m":
            return 2
        elif category == "t":
            return 3

    x, y = [], []
    save_path = "../data/na/split_na.pkl"
    save_wv_matrix_path = "../data/na/split_mr_wv_matrix.pkl"
    data_path = os.path.join(PROJECT_ROOT, "data/na/uci-news-aggregator.csv")
    na_data = pd.read_csv(data_path)
    tiltes = na_data["TITLE"].dropna()
    labels = na_data["CATEGORY"][tiltes.index].apply(category2lable)
    labels = labels.values
    tiltes = tiltes.values

    tiltes, labels = shuffle(tiltes, labels)

    sub_index = len(tiltes) // 10

    sub_titles = tiltes[:sub_index]
    sub_labels = labels[:sub_index]

    for title, label in zip(sub_titles, sub_labels):
        title = clean_data_for_look(title)
        x.append(title.split())
        y.append(label)

    test_idx = len(x) // 10 * 8

    data = set_data(x, y, test_idx)
    wv_matrix = make_wv_matrix(data, word_vectors)

    save_pickle(save_path, data)
    save_pickle(save_wv_matrix_path, wv_matrix)


def divide_imdb(word_vectors):
    def read_imdb(data_set='train', max_len=10000):
        """
        Args:
            data_set:
            max_len:

        Returns:

        """
        x, y = [], []
        source_folder = "../data/imdb/aclImdb"
        train_neg_path = os.path.join(source_folder, data_set, 'neg')
        train_pos_path = os.path.join(source_folder, data_set, 'pos')
        neg_files = os.listdir(train_neg_path)
        pos_files = os.listdir(train_pos_path)
        pos_files, neg_files = shuffle(pos_files, neg_files)

        for file_name in pos_files:
            with open(os.path.join(train_pos_path, file_name), 'r', encoding="utf-8") as f:
                line = f.readline()
                line = clean_data_for_look(line)
                if len(line.split()) <= max_len:
                    x.append(line.split())
                    y.append(1)

        for file_name in neg_files:
            with open(os.path.join(train_neg_path, file_name), 'r', encoding="utf-8") as f:
                line = f.readline()
                line = clean_data_for_look(line)
                if len(line.split()) <= max_len:
                    x.append(line.split())
                    y.append(0)
        return x, y

    save_path = "../data/imdb/split_imdb.pkl"
    save_wv_matrix_path = "../data/imdb/split_imdb_wv_matrix.pkl"
    x, y = read_imdb("train")
    # x1, y1 = read_imdb("train")
    # x2, y2 = read_imdb("test")
    # x = x1 + x2
    # y = y1 + y2
    x, y = shuffle(x, y, random_state=2021)
    test_idx = len(x) // 10 * 8
    data = set_data(x, y, test_idx)
    wv_matrix = make_wv_matrix(data, word_vectors)
    save_pickle(save_path, data)
    save_pickle(save_wv_matrix_path, wv_matrix)
    print("Total:", len(x), "train:", test_idx)


def _make_data(dataset):
    word2vec_model_path = "../data/word_vectors/glove_word2vec_f.txt"
    print('loading word2vector model....')
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_path, binary=False)
    if dataset == "all" or dataset == DataCorpus.NA:
        print("divide na....")
        divide_na(word2vec_model)
    if dataset == "all" or dataset == DataCorpus.MR:
        print("divide mr....")
        divide_mr(word2vec_model)
    if dataset == "all" or dataset == DataCorpus.IMDB:
        print("divide imdb....")
        divide_imdb(word2vec_model)

if __name__ == '__main__':
    _dataset = sys.argv[1]
    _make_data(_dataset)

