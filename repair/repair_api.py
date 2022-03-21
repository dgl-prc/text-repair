import sys

sys.path.append("../")
import os
import logging
from detect.diff_test import *
from utils.help_func import save_pickle, load_pickle
from repair.repairs_factory import *
from repair.sprt import *
from utils.constant import *
from utils.time_util import current_timestamp
import tqdm
import time

def print_info(p, repair_results):
    pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst = repair_results["pass_diff"], repair_results["success_cnt1"], \
                                                          repair_results["success_cnt2"], repair_results[
                                                              "sprt_cnt_lst"]
    print(
        "\n{} Repaired:{}, num_pass_diff:{}, success_cnt1:{}, success_cnt2:{}, total success:{:.2f}%, Avg sprt:{:.2f}".format(
            current_timestamp(),
            p,
            pass_diff,
            success_cnt1,
            success_cnt2,
            100 * ((success_cnt1 + success_cnt2) / p),
            np.mean(sprt_cnt_lst)
        ))


def print_small_size_repair_info(data_type, target_md, p, pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst):
    if data_type == DataCorpus.NA:
        if target_md == ModelType.TEXT_CNN and p == 99:
            print_info(p, pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst)
        if target_md == ModelType.LSTM1 and p == 98:
            print_info(p, pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst)
    elif data_type == DataCorpus.MR:
        if target_md == ModelType.TEXT_CNN and p == 273:
            print_info(p, pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst)
        if target_md == ModelType.LSTM1 and p == 279:
            print_info(p, pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst)
    elif data_type == DataCorpus.IMDB:
        if target_md == ModelType.TEXT_CNN and p == 279:
            print_info(p, pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst)
        if target_md == ModelType.LSTM1 and p == 296:
            print_info(p, pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst)


def process_repair_result(sprt_repair, status, repaired_label, y_ture, sprt_cnt, repair_results):
    if sprt_repair.is_pass_diff:
        repair_results["pass_diff"] += 1
    if repaired_label == y_ture:
        repair_results["sprt_cnt_lst"].append(sprt_cnt)
        if status == RepairStatus.REPAIR_SUCCESS:
            repair_results["success_cnt1"] += 1
        else:
            repair_results["success_cnt2"] += 1


def save_check_point(current_p, last_p, step, tmp_path, check_point_data, repair_results):
    if current_p % step == 0 and current_p > 0:
        save_pickle(os.path.join(tmp_path, "check_point_{}.pkl".format(current_p)),
                    {"candas": check_point_data, "repair_info": repair_results})
        last_check_point = os.path.join(tmp_path, "check_point_{}.pkl".format(last_p))
        if os.path.exists(last_check_point):
            os.remove(last_check_point)
        print_info(current_p, repair_results)


def load_check_point(last_p, check_point_path):
    check_point = load_pickle(os.path.join(check_point_path, "check_point_{}.pkl".format(last_p)))
    candas = check_point["candas"]
    repair_results = check_point["repair_info"]
    return candas, repair_results


def repair_sprt(detector, adv_data, repair_type, sprt_args, check_point_path, max_test_size, check_point=-1,word2vec=None):
    """

    Args:
        data:  adversarial data
        data_type:
        repair_type:
        sprt_args:
        target_md:  target model
        detector_mt1:
        detector_mt2:
        middle_data_path:
        kl_t:
        max_test_size:

    Returns:

    """
    sprt_repair = CoreSPRT(detector, sprt_args["rho"], sprt_args["alpha"], sprt_args["beta"],
                           sprt_args["extend_scale"], sprt_args["relax_scale"])
    repair_worker = load_repair_worker(repair_type, clfs=detector.classifier_array, eps=detector.epsilon,word2vec=word2vec)
    #############
    # statistics
    #############
    repair_results = {}
    if check_point == -1:
        p = 0
        last_p = 0
        generated_repairs = {}
        repair_results["pass_diff"] = 0
        repair_results["success_cnt1"] = 0  # get right label via sprt
        repair_results["success_cnt2"] = 0  # get right label via vote
        repair_results["majority_vote"] = 0  # vote directly, without sprt
        repair_results["sprt_cnt_lst"] = []
    else:
        repaired_candas, repair_results = load_check_point(check_point, check_point_path)
        p = check_point
        last_p = check_point
        generated_repairs = repaired_candas
    print(f"{current_timestamp()} Begin repair...")
    stime = time.time()
    for x, y_true in tqdm.tqdm(zip(adv_data["x"][p:], adv_data["y"][p:]), total=max_test_size, initial=p):
        candas = repair_worker.generate_repairs(x)
        generated_repairs[" ".join(x)] = candas
        status, repaired_label, sprt_cnt = sprt_repair.hypothesis_tesing(x, candas)
        process_repair_result(sprt_repair, status, repaired_label, y_true
                              , sprt_cnt, repair_results)
        p += 1
        save_check_point(p, last_p, 10, check_point_path, generated_repairs, repair_results)
        last_p = p
        ##############################
        # report result with small size
        ##############################
        # print_small_size_repair_info(data_type, target_md, p, pass_diff, success_cnt1, success_cnt2, sprt_cnt_lst)
        if p >= max_test_size:
            save_check_point(p, last_p, 1, check_point_path, generated_repairs, repair_results)
            break
    print(f">>>>>>>>>>>>>>>>>Avg time: {(time.time() - stime) / p}")
    return repair_results


def repair_online(detector, adv_data, repair_type, sprt_args, check_point_path, max_test_size, check_point):
    sprt_repair = CoreSPRT(detector, sprt_args["rho"], sprt_args["alpha"], sprt_args["beta"],
                           sprt_args["extend_scale"], sprt_args["relax_scale"])
    repair_worker = load_repair_worker(repair_type, clfs=detector.classifier_array, eps=detector.epsilon, max_depth=1)
    if check_point > 0:
        repaired_candas, repair_results = load_check_point(check_point, check_point_path)
        p = check_point
        last_p = check_point
        repair_worker.translated_texts = repaired_candas
    else:
        repair_results = {}
        repair_results["pass_diff"] = 0
        repair_results["success_cnt1"] = 0  # get right label via sprt
        repair_results["success_cnt2"] = 0  # get right label via vote
        repair_results["majority_vote"] = 0  # vote directly, without sprt
        repair_results["sprt_cnt_lst"] = []
        p = 0
        last_p = 0
        # use the translated texts if it exits.
        if os.path.exists(os.path.join(check_point_path, "check_point_{}.pkl".format(last_p))):
            print("Loading translated texts...")
            repaired_candas, _ = load_check_point(p, check_point_path)
            repair_worker.translated_texts = repaired_candas
        else:
            print("Repair from scratch....")
    print("Begin repair....")
    for y_true, y_adv, adv_text in tqdm.tqdm(zip(adv_data["y"][p:], adv_data["adv_y"][p:], adv_data["x"][p:]),
                                             total=max_test_size, initial=p):
        adv_node = Paraphrase(" ".join(adv_text))
        candidates = []
        for trans_texts in repair_worker.bfs_parapharase(adv_node):
            candidates.extend(trans_texts)
            status, repaired_label, sprt_cnt = sprt_repair.hypothesis_tesing(adv_text, candidates)
            if repaired_label == y_true:
                process_repair_result(sprt_repair, status, repaired_label, y_true
                                      , sprt_cnt, repair_results)
                break
        p += 1
        save_check_point(p, last_p, 1, check_point_path, repair_worker.translated_texts, repair_results)
        last_p = p
        if p >= max_test_size:
            break
            save_check_point(p, last_p, 1, check_point_path, repair_worker.translated_texts, repair_results)
    return repair_results
