"""
This file contains several metrics but only the kld is finally adopted.
"""
from math import log as ln
import numpy as np
import collections


class MetricType:
    AJC = "ajc"
    KLD = "kld"
    RBO = "RBO"
    LD = "ld"


def get_statictics(data_seq):
    size = len(data_seq)
    avg = np.average(data_seq)
    std = np.std(data_seq)
    c95 = 1.96 * std / np.sqrt(size)  # 95%
    c98 = 2.33 * std / np.sqrt(size)  # 98%
    c99 = 2.58 * std / np.sqrt(size)  # 99%
    confidence = {"c95": c95, "c98": c98, "c99": c99}
    return {"avg": avg, "std": std, "confidence": confidence}


def kld(P, Q):
    """ calculate Kl-divergence
    Args:
        P: the probability vector of model1
        Q: the probability vector of model2
    Returns:
        Kl-divergence
    """
    assert len(P) == len(Q)
    kl = 0.
    for i in range(len(P)):
        kl += P[i] * ln(Q[i] / P[i])
    return -1 * kl


# @DeprecationWarning
def ajc(preds1, preds2):
    """
    Average jaccard  Coefficient
    Args:
        preds1:
        preds2:
    Returns:
    """
    assert len(preds1) == len(preds2)
    k = len(preds1)
    jaccard = 0.
    for i in range(k):
        set1 = set(preds1[:i + 1])
        set2 = set(preds2[:i + 1])
        jaccard += len(set1 & set2) / len(set1 | set2)
    return jaccard / k


# @DeprecationWarning
def ld(pred_seqs):
    '''
    label difference.
    :param pred_seqs: the labels list of all classifiers on the same input
    :return:
    '''
    a = len(list(dict(collections.Counter(pred_seqs)).items()))
    if a == len(pred_seqs):
        return True  # not unanimous, suspicious samples
    return False  # unanimous, normal samples
