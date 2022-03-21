import sys

sys.path.append("../")
import os
import gensim
from utils.help_func import load_pickle
from utils.constant import *


def load_word2vec(init_sims=False):
    word2vec_path = os.path.join(PROJECT_ROOT, WORD2VEC)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    word2vec.init_sims(replace=init_sims)
    return word2vec


def select_normal_data(data, wl_idx_list):
    X = []
    Y = []
    idx_list = [i for i in range(len(data["test_x"])) if i not in wl_idx_list]
    for idx in idx_list:
        X.append(data["test_x"][idx])
        Y.append(data["test_y"][idx])
    return {"x": X, "y": Y}


def read_mr():
    data_path = os.path.join(PROJECT_ROOT, "data/mr/split_mr.pkl")
    return load_pickle(data_path)


def wv_matrix_mr():
    matrix_path = os.path.join(PROJECT_ROOT, "data/mr/split_mr_wv_matrix.pkl")
    return load_pickle(matrix_path)


def make_dtd_data(data, dtd_idx, frm, to):
    dtd_data = {}
    if to == -1:
        dtd_data["x"] = [data["test_x"][i] for i in dtd_idx]
        dtd_data["y"] = [data["test_y"][i] for i in dtd_idx]
        dtd_data["idx"] = dtd_idx
    else:
        dtd_data["x"] = [data["test_x"][i] for i in dtd_idx[frm:to]]
        dtd_data["y"] = [data["test_y"][i] for i in dtd_idx[frm:to]]
        dtd_data["idx"] = dtd_idx[frm:to]
    return dtd_data


def read_dtd_mr(model_type, frm=0, to=-1):
    data = read_mr()
    dtdpath = os.path.join(PROJECT_ROOT, DTD.MR).format(model_type)
    dtd_idx = load_pickle(dtdpath)
    return make_dtd_data(data, dtd_idx, frm, to)


def read_dtd_na(model_type, frm=0, to=-1):
    data = read_na()
    dtdpath = os.path.join(PROJECT_ROOT, DTD.NA).format(model_type)
    dtd_idx = load_pickle(dtdpath)
    return make_dtd_data(data, dtd_idx, frm, to)


# def read_dtd_imdb(model_type, frm=0, to=-1, max_len=150):
#     data = read_imdb()
#     dtdpath = os.path.join(PROJECT_ROOT, DTD.IMDB).format(model_type)
#     dtd_idx = load_pickle(dtdpath)
#     dtd_data = make_dtd_data(data, dtd_idx, frm, to)
#     new_data = {"x": [], "y": [], "idx": []}
#     cnt = 0
#     for x, y, index in zip(dtd_data['x'], dtd_data['y'], dtd_data['idx']):
#         if len(x) <= max_len:
#             new_data["x"].append(x)
#             new_data["y"].append(y)
#             new_data["idx"].append(index)
#             cnt += 1
#         if cnt == to:
#             break
#     return new_data


# old fansion
def read_dtd_imdb(frm=0, to=-1, max_len=100):
    data = read_imdb()
    dtdpath = os.path.join(PROJECT_ROOT, DTD.IMDB)
    dtd_idx = load_pickle(dtdpath)
    dtd_data = make_dtd_data(data, dtd_idx, frm=0, to=-1)
    new_data = {"x": [], "y": [], "idx": []}
    cnt = 0
    for x, y, index in zip(dtd_data['x'], dtd_data['y'], dtd_data['idx']):
        if len(x) <= max_len:
            new_data["x"].append(x)
            new_data["y"].append(y)
            new_data["idx"].append(index)
            cnt += 1
        if cnt == to:
            break
    return new_data


def read_na():
    data_path = os.path.join(PROJECT_ROOT, "data/na/split_na.pkl")
    return load_pickle(data_path)


def wv_matrix_na():
    matrix_path = os.path.join(PROJECT_ROOT, "data/na/split_mr_wv_matrix.pkl")
    return load_pickle(matrix_path)


def read_imdb():
    data_path = os.path.join(PROJECT_ROOT, "data/imdb/split_imdb.pkl")
    return load_pickle(data_path)


def wv_matrix_imdb():
    matrix_path = os.path.join(PROJECT_ROOT, "data/imdb/split_imdb_wv_matrix.pkl")
    return load_pickle(matrix_path)


###################################
# use experiments.step2_search_threshold.search_metric_threshold.get_wl_data(data_type, model_type) instead.
####################################
# def read_wl_data(data_path):
#     data = {}
#     wl_data = load_pickle(data_path)
#     data["x"] = wl_data["wl_x"]
#     data["y"] = wl_data["wl_y"]
#     data["idx"] = wl_data["wl_idx"]
#     return data


def merge_data(adv_cnn, adv_lstm, pararent_data):
    '''
    Merge the redundant sentnece
    '''
    new_data = {}
    for sent, adv_label, idx in zip(adv_cnn["x"], adv_cnn["y"], adv_cnn["idx"]):
        if isinstance(sent, list):
            sent = " ".join(sent)
        new_data[sent] = {"y_cnn": adv_label, "y_true": pararent_data["test_y"][idx]}

    for sent, adv_label, idx in zip(adv_lstm["x"], adv_lstm["y"], adv_lstm["idx"]):
        if isinstance(sent, list):
            sent = " ".join(sent)
        if sent in new_data:
            assert pararent_data["test_y"][idx] == new_data[sent]["y_true"]
            new_data[sent]["y_lstm"] = adv_label
        else:
            new_data[sent] = {"y_lstm": adv_label, "y_true": pararent_data["test_y"][idx]}
    # change to the standard format
    std_data = {}
    sents = []
    Y = []
    Y_ture = []
    for key in new_data.keys():
        sents.append(key)
        Y_ture.append(new_data[key]["y_true"])
        adv_lables = {}
        if "y_cnn" in new_data[key]:
            adv_lables["y_cnn"] = new_data[key]["y_cnn"]
        if "y_lstm" in new_data[key]:
            adv_lables["y_lstm"] = new_data[key]["y_lstm"]
        Y.append(adv_lables)

    std_data["x"] = sents
    std_data["y"] = Y
    std_data["y_true"] = Y_ture
    return std_data


def load_adv_text(attack_type, target_model_type, dataset):
    adv_path = AdvData.get_adv_data_path(attack_type, target_model_type, dataset)
    adv_data = load_pickle(adv_path)
    ori_data = getattr(sys.modules[__name__], "read_{}".format(dataset))()
    y_ture = [ori_data["test_y"][idx] for idx in adv_data["idx"]]
    return y_ture, adv_data["y"], adv_data["x"]


if __name__ == '__main__':
    dataset = DataCorpus.IMDB
    dtd = read_dtd_imdb(to=300)
    print(len(dtd["x"]))
