"""
DTD: Docile Test Data. which means the target model could give a correct prediction.
here, we only save the index of data["test_x"]. and the index list is shuffled with random seed 20190807
Notice that since the idx are shuffled, we could directly choose top-k elements as a randomly
"""
import sys

sys.path.append("../")
import os
from utils.constant import *
from utils.help_func import save_pickle, load_pickle
import random
from utils import data_reader
from models.adapter import make_classifier


def make_dtd_index(dataset, model_type1, model_type2):
    '''
    :param dataset:
    :return: a dict with three keys: x,y,idx. Note that the idx is the idx in text data set.

    Args:
        model_type2:
        model_type1:
    '''
    classifier1 = make_classifier(model_type1, dataset)
    classifier2 = make_classifier(model_type2, dataset)
    data = getattr(data_reader, "read_{}".format(dataset))()
    indecies = []
    idx = 0
    for x, y in zip(data["test_x"], data["test_y"]):
        label1 = classifier1.get_label(x)
        label2 = classifier2.get_label(x)
        if label1 == label2 and label2 == y:
            indecies.append(idx)
        idx += 1
    random.seed(20190807)
    random.shuffle(indecies)
    save_path = os.path.join(PROJECT_ROOT, getattr(DTD, dataset.upper()))
    save_pickle(save_path, indecies)


def debug():
    """compare with two imdb processed dataset"""
    root = "/home/dgl/project/text_repair/data/imdb"
    # split_imdb.pkl
    amax_path = os.path.join(root, "split_imdb.pkl")
    ade_path = os.path.join(root, "data_ade/split_imdb.pkl")
    amax_data = load_pickle(amax_path)
    ade_data = load_pickle(ade_path)

    # compare test
    print(amax_data["test_x"] == ade_data["test_x"])
    print(amax_data["test_y"] == ade_data["test_y"])

    for x1, x2 in zip(amax_data["test_x"], ade_data["test_x"]):
        if x1 != x2:
            print(x1)
            print(x2)
            break


if __name__ == '__main__':
    _dataset = sys.argv[1]
    make_dtd_index(_dataset, ModelType.TEXT_CNN, ModelType.LSTM1)
    # debug()
