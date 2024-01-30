import numpy as np

def get_label_dist(labels, num_classes):
    pass


def pure_MPE_estimator(src_probs, tgt_probs, num_allowed_fp=0):
    sorted_src_probs = np.sort(src_probs)[::-1]
    sorted_src_probs_threshold = sorted_src_probs[num_allowed_fp]
    estimated_mpe = np.sum(tgt_probs > sorted_src_probs_threshold) / len(tgt_probs)
    return estimated_mpe, sorted_src_probs_threshold

def DKW_bound(x, y, t, m, n, delta=0.1, gamma=0.01):
    # Copied from https://github.com/acmi-lab/PU_learning/blob/main/estimator.py

    temp = np.sqrt(np.log(1/delta)/2/n) + np.sqrt(np.log(1/delta)/2/m)
    bound = temp*(1+gamma)/(y/n)

    estimate = t

    return estimate, t - bound, t + bound

def top_bin_estimator(pdata_probs, udata_probs):

def BBE_estimate_binary(src_probs, tgt_probs):
    pass


def estimator_CM_EN():
    pass
