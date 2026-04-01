import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def compute_per_class_metrics(y_true: List[int], y_pred: List[int], labels: List[str]) -> Dict:
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i])
        }
    return per_class


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: List[int], y_pred: List[int], labels: List[str]) -> str:
    return classification_report(y_true, y_pred, target_names=labels, zero_division=0)
