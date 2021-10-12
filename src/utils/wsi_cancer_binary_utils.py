import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score, \
    recall_score, precision_score

def calc_wsi_cancer_binary_metrics(prediction_confidences, gt):
    # TODO: calc metric and cofusion matrices
    metrics = {}
    prediction_confidences = prediction_confidences[...,1]
    predictions = prediction_confidences > 0.5
    gt = np.argmax(gt, axis=-1)

    metrics['wsi_accuracy'] = accuracy_score(gt, predictions)
    metrics['wsi_cohens_quadratic_kappa'] = cohen_kappa_score(gt, predictions, weights='quadratic')
    metrics['wsi_f1_score'] = f1_score(gt, predictions)
    metrics['wsi_recall'] = recall_score(gt, predictions)
    metrics['wsi_precision'] = precision_score(gt, predictions)
    metrics['wsi_auc'] = roc_auc_score(gt, prediction_confidences)
    roc = {}
    roc['fpr'], roc['tpr'], roc['thresholds'] = roc_curve(gt, prediction_confidences)
    artifacts = {}
    artifacts['roc'] = roc

    conf_matrix = confusion_matrix(gt,
                                   predictions,
                                   labels=[0, 1])

    return metrics, conf_matrix