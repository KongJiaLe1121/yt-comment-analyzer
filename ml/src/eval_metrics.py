import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

def sentiment_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted")
    }

def print_report(y_true, y_pred, labels=None):
    print(classification_report(y_true, y_pred, target_names=labels))
