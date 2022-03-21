import sys

sys.path.append("../../")
import numpy as np
from experiments.repairs_generation.paraphraser import PLACE_HOLDER
from text_attack.translation_based.tranlator_attack import BaiDuTranslatorAttacker
from detect.diff_test import DiffType
from utils.constant import *
from utils.exceptions import *

# support_langs = BaiDuTranslatorAttacker(None).get_all_support_lang()

class TestMode:
    ONLINE = 0
    OFFLINE = 1


def filter_invalid_paraphrase(input_text, paraphrases):
    new_input_text = input_text.strip().lower()
    keys = [key for key in paraphrases]
    for para in keys:
        if new_input_text == para.strip().lower():
            del paraphrases[para]


def node_inspect(adv_node):
    num_children = len(adv_node.children)
    print("num of children:{}".format(num_children))
    for pivot in support_langs:
        if pivot in adv_node.children:
            print("descendants of childre {}:{}".format(pivot, len(adv_node.children[pivot].children)))


def one_depth_inspect(repairs1, repairs2, adv_text):
    onedepth1 = repairs1[adv_text]
    onedepth2 = repairs2[adv_text]

    print("onedepth1:{},onedepth2:{}".format(len(onedepth1), len(onedepth2)))
    print(onedepth1)
    print(onedepth2.keys())





class CoreSPRT():

    def __init__(self, diffJudge, rho=0.8, alpha=0.1, beta=0.1, extend_scale=1.0, relax_scale=0.2,
                 show_performance=False):
        self.removed_label = {}
        self.init_bad_label = -1  # the label of adversary
        self.diffJudge = diffJudge
        self.label_fre = {}
        self.input_flow = []
        self.removed_posistion = []
        self.is_pass_diff = False
        self.show_performance = show_performance
        self.alpha = alpha
        self.beta = beta
        self.acc_pr = np.log(beta / (1 - alpha))
        self.deny_pr = np.log((1 - beta) / alpha)
        self.p0, self.p1 = self.get_threshold_relax(rho, extend_scale, relax_scale)

    def get_threshold_relax(self, threshold, extend_scale=1.0, relax_scale=0.2):
        new_threshold, gamma = threshold * extend_scale, threshold * relax_scale
        p0, p1 = new_threshold - gamma, new_threshold + gamma
        return p0, p1

    def calculate_sprt_pr(self, label_fre, target_label):
        n = sum([label_fre[label] for label in label_fre.keys()])
        c = label_fre[target_label]
        return c * np.log(self.p0 / self.p1) + (n - c) * np.log((1 - self.p0) / (1 - self.p1))

    def reset(self):
        self.label_fre = {}
        self.removed_label = {}
        self.input_flow = []
        self.removed_posistion = []
        self.is_pass_diff = False
        self.init_bad_label = -1

    def init(self, adv_text):
        is_adv4cnn, pred_cnn, diffType_cnn = self.diffJudge.need_repair(adv_text)  # detect
        if diffType_cnn == DiffType.SUSPICIOUS_PATTERN:
            self.removed_label[pred_cnn] = 0
            self.init_bad_label = pred_cnn

    def hypothesis_tesing(self, adv_text, candas):
        self.reset()
        self.init(adv_text)
        trans_cnt = 0
        labels = [-1] * 650
        label_sprt = {0: [], 1: [], 2: [], 3: []}
        # if self.show_performance:
        #     print("acc_pr:{},deny_pr:{}".format(self.acc_pr, self.deny_pr))
        for i, cadn in enumerate(candas):
            if self.show_performance:
                sprt_rst = {l: label_sprt[l][-1] for l in self.label_fre}
                print(
                    "trans_cnt:{} check_length:0-{},1-{},2-{},3-{} label_freq:{},removed_freq:{},label_sprt={}".format(
                        trans_cnt, len(label_sprt[0]),
                        len(label_sprt[1]), len(label_sprt[2]),
                        len(label_sprt[3]),
                        self.label_fre,
                        self.removed_label,
                        sprt_rst
                        ))
            trans_cnt += 1
            if cadn == PLACE_HOLDER:
                # if self.show_performance:
                #     #####################
                #     # invalid text update
                #     ####################
                #     for l in [0, 1, 2, 3]:
                #         if len(label_sprt[l]) == 0:
                #             label_sprt[l].append(0)
                #         else:
                #             label_sprt[l].append(label_sprt[l][-1])
                continue
            try:
                need_repair, pred, diffType = self.diffJudge.need_repair(cadn)
                # if self.show_performance:
                #     labels[i] = pred
            except BadInput as e:
                print("Warning! Bad input found:{}".format(str(e)))
                continue
            if pred in self.removed_label:
                self.input_flow.append(pred)
                self.removed_label[pred] += 1
                # if self.show_performance:
                #     #####################
                #     # removed text update
                #     ####################
                #     for l in [0, 1, 2, 3]:
                #         if len(label_sprt[l]) == 0:
                #             label_sprt[l].append(0)
                #         else:
                #             label_sprt[l].append(label_sprt[l][-1])
                continue
            if not need_repair:
                self.is_pass_diff = True
                self.input_flow.append(pred)
                if pred in self.label_fre.keys():
                    self.label_fre[pred] += 1
                else:
                    self.label_fre[pred] = 1

                pr = self.calculate_sprt_pr(self.label_fre, pred)

                ################
                # normal update
                ################
                label_sprt[pred].append(pr)
                for l in [0, 1, 2, 3]:
                    if l != pred:
                        if len(label_sprt[l]) == 0:
                            label_sprt[l].append(0)
                        elif l in self.label_fre:
                            new_sprt = self.calculate_sprt_pr(self.label_fre, l)
                            label_sprt[l].append(new_sprt)
                        else:
                            label_sprt[l].append(label_sprt[l][-1])
                if pr <= self.acc_pr:
                    # if self.show_performance:
                    #     print("============labels================")
                    #     for l in labels[:trans_cnt]:
                    #         print(l)
                    #     print(">>>>>>>>>>>>>>>>cost:{}".format(trans_cnt))
                    #     for l in [0, 1, 2, 3]:
                    #         print(">>>>>>SPRT of label-{}<<<<<<<<<<<<".format(l))
                    #         for ration in label_sprt[l]:
                    #             print(ration)
                    #         print("=======================")
                    #     print("final:sprt {}".format(pr))
                    # print("SUCCESS init-remove:{},removed_labels:{},label_freq:{}".format(self.init_bad_label,self.removed_label,self.label_fre))
                    return RepairStatus.REPAIR_SUCCESS, pred, trans_cnt
                elif pr >= self.deny_pr:
                    ######################################
                    # remove label "pred" from label_fre
                    #####################################
                    self.removed_label[pred] = self.label_fre[pred]
                    del self.label_fre[pred]
                    for l in [0, 1, 2, 3]:
                        if l in self.label_fre:
                            new_sprt = label_sprt[l][-1]
                            if new_sprt > self.deny_pr:
                                self.removed_label[l] = self.label_fre[l]
                                del self.label_fre[l]
                                print("remove label:{}".format(l))

            elif self.show_performance:
                #####################
                # invalid text update
                ####################
                for l in [0, 1, 2, 3]:
                    if len(label_sprt[l]) == 0:
                        label_sprt[l].append(0)
                    else:
                        label_sprt[l].append(label_sprt[l][-1])

        ###########
        #  vote
        ###########
        items = [item for item in self.label_fre.items()]
        if len(items) > 0:
            self.vote_pred = sorted(items, key=lambda x: x[1], reverse=True)[0][0]
        else:
            self.vote_pred = -1
        return RepairStatus.NO_BESTLABEL, self.vote_pred, trans_cnt

