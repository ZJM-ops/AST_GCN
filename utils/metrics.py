import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, roc_auc_score)


def compute_metrics(labels, preds, probs=None, num_classes=3):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)

    if probs is not None:
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')
    else:
        auc = float('nan')

    cm = confusion_matrix(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'confusion_matrix': cm
    }