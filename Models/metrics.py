import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, confusion_matrix, balanced_accuracy_score
)

def compute_all_metrics(y_true, y_pred, average="macro", labels=None, output_dict=False):
    out = {}
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    out["f1_" + average] = f1_score(y_true, y_pred, average=average)
    out["precision_" + average] = precision_score(y_true, y_pred, average=average, zero_division=0)
    out["recall_" + average] = recall_score(y_true, y_pred, average=average, zero_division=0)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)

    if output_dict:
        return out
    else:
        for k, v in out.items():
            print(k, ":\n", v)
        return out