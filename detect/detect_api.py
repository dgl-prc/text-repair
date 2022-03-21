import random
from detect.diff_test import *
import random
from detect.diff_test import *
from utils.data_reader import load_adv_text
from utils.data_reader import select_normal_data
from utils.constant import PROJECT_ROOT, SavedWLPath
from utils.help_func import load_pickle
from utils import data_reader
import os
import time


def prepare_data(dataset, model_type, attack_type):
    _, _, adv_data = load_adv_text(attack_type, model_type, dataset)
    wl_path = os.path.join(PROJECT_ROOT, SavedWLPath.format(dataset, model_type))
    wl_idx_list = load_pickle(wl_path)
    data = getattr(data_reader, "read_{}".format(dataset))()
    normal_texts = select_normal_data(data, wl_idx_list)
    return normal_texts["x"], adv_data


def do_detect(data_type, attack_type, kl_t, target_model_type, detector_mt1, detector_mt2, adv_size=500,
              is_vanilla_df=False):
    """
    Args:
        data_type:
        attack_type:
        kl_t:
        target_model_type:  target_model_type
        detector_mt1: detector model type1
        detector_mt2: detector model type2
        is_vanilla_df:
    Returns:
    """
    model1 = make_classifier(detector_mt1, data_type)
    model2 = make_classifier(detector_mt2, data_type)
    detector = [model1, model2]
    test_data = {}
    normal_texts, adv_texts = prepare_data(data_type, target_model_type, attack_type)
    adv_texts = adv_texts[:adv_size]
    rnd_idcies = [i for i in range(len(normal_texts))]
    # random.seed(20200202)
    random.seed(20210202)
    random.shuffle(rnd_idcies)
    rnd_idcies = rnd_idcies[:len(adv_texts)]
    test_data["x"] = adv_texts + [normal_texts[idx] for idx in rnd_idcies]
    test_data["y"] = [1] * len(adv_texts) + [0] * len(adv_texts)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    print(f"Detect with vanilla differenital testing:{is_vanilla_df}")
    stime = time.time()
    for x, y in zip(test_data["x"], test_data["y"]):
        if is_vanilla_df:
            pass_df = v_df_test(x, detector)
        else:
            pass_df, k, (pred1, prob1), (pred2, prob2) = df_test(x, detector, kl_t)
        if pass_df:  # not adv
            if y == 0:
                TN += 1
            else:
                FN += 1
        else:  # adv
            if y == 0:
                FP += 1
            else:
                TP += 1
    total = time.time() - stime
    print("{} {}>>{}+{} TP:{}, FP:{}, TN:{}, FN:{}".format(data_type, target_model_type, detector_mt1, detector_mt2, TP,
                                                           FP, TN, FN))
    acc = round(100 * (TP + TN) / len(test_data["x"]))
    fdr = round(100 * FP / (TP + FP))
    dr = round(100 * TP / (FN + TP))
    avg_time = 1000*total / len(test_data["x"])
    assert (FN + TP) == int(len(test_data["x"]) / 2)
    assert (FP + TN) == int(len(test_data["x"]) / 2)
    print("{} {}>>{}+{} accuracy:{}%, dr:{}% ,fdr:{}% avg_time:{} ms".format(data_type, target_model_type, detector_mt1,
                                                                          detector_mt2,
                                                                          acc, dr, fdr, avg_time))
