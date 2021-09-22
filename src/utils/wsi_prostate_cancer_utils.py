import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, accuracy_score

def calc_wsi_prostate_cancer_metrics(gt, predictions):
    predictions = np.argmax(predictions, axis=-1)
    gt = np.argmax(gt, axis=-1)
    metrics_dict = {}
    metrics_dict['test_cohens_quadratic_kappa'] = cohen_kappa_score(gt, predictions, weights='quadratic')
    metrics_dict['test_f1_score'] = f1_score(gt, predictions, average='macro')
    metrics_dict['test_accuracy'] = accuracy_score(gt, predictions)
    # confusion_matrices = {}
    #
    # confusion_matrices['wsi_isup_confusion_matrix'] = confusion_matrix(gt,
    #                                                                    predictions,
    #                                                                    labels=[0, 1, 2, 3, 4, 5])
    # artifacts = {}
    # artifacts['confusion_matrics'] = confusion_matrices

    conf_matrix = confusion_matrix(gt,
                                   predictions,
                                   labels=[0, 1, 2, 3, 4, 5])

    return metrics_dict, conf_matrix


def get_gleason_score_and_isup_grade(wsi_df):
    gleason_primary = np.array(wsi_df['Gleason_primary'])
    gleason_secondary = np.array(wsi_df['Gleason_secondary'])
    wsi_df['gleason_score'] = gleason_primary + gleason_secondary
    isup_grade = np.full(shape=len(wsi_df), fill_value=-1)
    isup_grade = np.where(np.logical_and(gleason_primary == 0, gleason_secondary == 0), 0, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 3, gleason_secondary == 3), 1, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 3, gleason_secondary == 4), 2, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 4, gleason_secondary == 3), 3, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 3, gleason_secondary == 5), 4, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 4, gleason_secondary == 4), 4, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 5, gleason_secondary == 3), 4, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 4, gleason_secondary == 5), 5, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 5, gleason_secondary == 4), 5, isup_grade)
    isup_grade = np.where(np.logical_and(gleason_primary == 5, gleason_secondary == 5), 5, isup_grade)
    assert(np.all(isup_grade >= 0))

    wsi_df['isup_grade'] = isup_grade

    return wsi_df


def calc_gleason_grade(num_predictions_per_class, confidences_per_class, confidence_threshold):
    # don't count outliers, set count to zero if top confidences are low
    # for i in range(len(num_predictions_per_class)):
    #     if confidences_per_class[i] < confidence_threshold:
    #         num_predictions_per_class[i] = 0
    if num_predictions_per_class[1] == num_predictions_per_class[2] == num_predictions_per_class[3] == 0:
        primary = 0
        secondary = 0
    # only one gleason grade
    elif num_predictions_per_class[2] == num_predictions_per_class[3] == 0:
        primary = 3
        secondary = 3
    elif num_predictions_per_class[1] == num_predictions_per_class[3] == 0:
        primary = 4
        secondary = 4
    elif num_predictions_per_class[1] == num_predictions_per_class[2] == 0:
        primary = 5
        secondary = 5
    # two gleason grades
    else:
        # Make sure we don't include class 0 here. Argsort returns value 0,1 or 2. Add 3 to get gleason grade.
        primary = np.argsort(num_predictions_per_class[1:4])[2] + 3
        secondary = np.argsort(num_predictions_per_class[1:4])[1] + 3

    return primary, secondary



