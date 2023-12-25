import numpy as np
import torch
import torch.nn as nn

def recall_from_logits(logits, targets, recall_label=1):
    assert logits.shape[1] == 2 # handle binary classification so far
    indicator_proxy = nn.Softmax(dim=1)
    preds = torch.argmax(logits, dim=1)
    label_counts = torch.bincount(targets.flatten())
    num_labels = logits.shape[1]
    num_positives = torch.Tensor([torch.sum(torch.multiply(preds==l, targets==l)) for l in range(num_labels)]) # tp
    recall = num_positives[recall_label] / label_counts[recall_label] # tp / p

    recall_loss = indicator_proxy(logits)[:, 1 - recall_label] # if high prob assign to recall label, low loss value
    recall_loss = torch.sum(recall_loss[targets==recall_label]) / label_counts[recall_label] # only look into the p's

    recall_proxy = 1 - recall_loss
    return recall, recall_proxy, recall_loss

def fpr_from_logits(logits, targets, positive_label=1):
    assert logits.shape[1] == 2 # handle binary classification so far
    indicator_proxy = nn.Softmax(dim=1)
    preds = torch.argmax(logits, dim=1)
    label_counts = torch.bincount(targets.flatten())
    num_labels = logits.shape[1]

    tn = torch.logical_and(preds == 1 - positive_label, targets == 1 - positive_label).sum()
    fp = torch.logical_and(preds == positive_label, targets == 1 - positive_label).sum()
    fpr = fp / (fp + tn)

    positive_scores = indicator_proxy(logits)[:, positive_label]
    fpr_proxy = positive_scores[targets!=positive_label].sum() / (fp + tn)
    return fpr, fpr_proxy