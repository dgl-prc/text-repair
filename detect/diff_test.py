import numpy as np
from detect.metric import kld
from utils.exceptions import BadInput
from experiments.step2_search_threshold.search_metric_threshold import get_threshold
from models.adapter import make_classifier


class DiffType:
    LABEL_DIVERSE = "label_diverse"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    PASS_DIFF = "pass_diff"


def kld_pred(sent, clfs):
    """ calculate the kl-divergence with cnn and lstm classifiers
    Args:
        sent (list or str): a sentence
        clfs (tuple): (classifier1,classifier2)
    Returns:
        k (float): the kl-divergence of two output distribution
        result_clf1 (tuple): (pred, confidence)
        result_clf2 (tuple): (pred, confidence)
    """
    if isinstance(sent, str):
        sent = sent.split()
    probs1 = clfs[0].get_probs(sent)
    probs2 = clfs[1].get_probs(sent)
    pred1 = np.argmax(probs1)
    pred2 = np.argmax(probs2)
    k = kld(probs1, probs2)
    return k, (pred1, probs1[pred1]), (pred2, probs2[pred2])


def v_df_test(sent, clfs):
    """vanilla differential testing.
      only compare the predicts (labels) of two models.
    Args:
        sent (list or str): a sentence
        clfs (tuple): (classifier1,classifier2)
    Returns:
        pass_df (bool): True if the two predicts are same or False
    """
    k, (pred1, prob1), (pred2, prob2) = kld_pred(sent, clfs)
    pass_df = True
    if pred1 != pred2:
        pass_df = False
    return pass_df


def df_test(sent, clfs, kl_t):
    """ differential testing with kl-divergence
    Args:
        sent (list or str): a sentence
        clfs (tuple): (classifier1,classifier2)
        kl_t (float): the threshold of kl-divergence.
                      one is deemed malicious when its kld exceeds this threshold
    Returns:
        pass_df (bool): True if the two predicts are same or False
        k (float): the kl-divergence of two output distribution
        result_clf1 (tuple): (pred, confidence)
        result_clf2 (tuple): (pred, confidence)
    """
    assert len(clfs) == 2
    k, (pred1, prob1), (pred2, prob2) = kld_pred(sent, clfs)
    pass_df = True
    if pred1 != pred2:
        pass_df = False
    elif k >= kl_t:
        pass_df = False
    return pass_df, k, (pred1, prob1), (pred2, prob2)


class DiffJudge(object):
    def __init__(self, clfs, kl_t):
        """
        Args:
            clfs (tuple): (classifier1,classifier2)
            kl_t (float): the threshold of kl-divergence.
                          one is deemed malicious when its kld exceeds this threshold
        """
        self.classifier_array = clfs
        self.epsilon = kl_t

    def get_probs(self, x):
        probs1 = self.classifier_array[0].get_probs(x)
        probs2 = self.classifier_array[1].get_probs(x)
        return probs1, probs2

    def need_repair(self, x):
        try:
            is_benign, kld, (pred1, prob1), (pred2, prob2) = df_test(x, self.classifier_array, self.epsilon)
        except BadInput as e:
            raise BadInput(x)
        if is_benign:
            return False, pred1, DiffType.PASS_DIFF
        else:
            if pred1 != pred2:
                return True, -1, DiffType.LABEL_DIVERSE
            else:
                return True, pred1, DiffType.SUSPICIOUS_PATTERN


def make_detector(data_type, target_md, detector_mt1, detector_mt2, kl_t=-1.):
    if kl_t < 0:
        print("init kl threshold...")
        kl_t_tuple = get_threshold(data_type, detector_mt1, detector_mt2)
        if target_md == detector_mt1:
            kl_t = kl_t_tuple[0]
        elif target_md == detector_mt2:
            kl_t = kl_t_tuple[1]
        else:
            kl_t = sum(kl_t_tuple) / 2
    model1 = make_classifier(detector_mt1, data_type)
    model2 = make_classifier(detector_mt2, data_type)
    detector = [model1, model2]
    diffJudge = DiffJudge(detector, kl_t)
    return diffJudge
