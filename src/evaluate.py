import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.wsi_prostate_cancer_utils import calc_wsi_prostate_cancer_metrics
from utils.saving import save_metrics_and_conf_matrics

def att_evaluation(instance_model, test_gen):
    n = len(test_gen)
    correct_attention_preds = 0
    correct_attention_std = 0
    wrong_attention_preds = 0
    wrong_attention_std = 0

    pos_bags = 0
    for i in range(n):
        x_bag, y_bag = test_gen.images[i], test_gen.labels[i]
        # bag_pred = bag_model.predict(x_bag)
        preds = instance_model.predict(x_bag)
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        correct_attention, att_std = _correct_att_prediction(mean, std, y_bag)
        if correct_attention:
            correct_attention_preds += 1
            correct_attention_std += att_std
        else:
            wrong_attention_preds += 1
            wrong_attention_std += att_std

        pos_bags += np.max(y_bag)
    attention_accuracy = correct_attention_preds/pos_bags
    correct_att_std = correct_attention_std/correct_attention_preds
    wrong_att_std = wrong_attention_std/wrong_attention_preds

    print('Attention accuracy: ',  str(attention_accuracy))
    print('Correct attention std: ',  str(correct_att_std))
    print('Wrong attention std: ',  str(wrong_att_std))

def _correct_att_prediction(mean, std, instance_gt):
    am = np.argmax(mean)
    am_std = std[am]
    if instance_gt[am] == 1:
        return 1, am_std
    else:
        return 0, am_std

def bag_level_evaluation(test_gen, bag_level_uncertainty_model: tf.keras.Model, out_dir):
    """
    Evaluate performance on bag level (predictions and uncertainty estimation)
    """
    os.makedirs(out_dir, exist_ok=True)
    std_threshold = 0.02

    n = len(test_gen)
    correct_preds = 0
    correct_stds = np.array([])
    wrong_preds = 0
    wrong_stds = np.array([])
    preds = bag_level_uncertainty_model.predict(test_gen)

    mean_below_threshold = np.array([])
    gt_below_threshold = np.array([])

    for i in range(n):
        y_bag = test_gen.labels[i]
        bag_pred = preds[i]
        mean = np.mean(bag_pred, axis=0)
        std = np.std(bag_pred, axis=0)
        pred_class = np.argmax(mean)

        correct_pred = (pred_class == np.argmax(y_bag))
        if correct_pred:
            correct_preds += 1
            correct_stds = np.append(correct_stds, np.mean(std))
        else:
            wrong_preds += 1
            wrong_stds = np.append(wrong_stds, np.mean(std))
        if np.mean(std) <= std_threshold:
            if i == 0:
                mean_below_threshold = np.expand_dims(mean, axis=0)
                gt_below_threshold = np.expand_dims(y_bag, axis=0)
            else:
                mean_below_threshold = np.append(mean_below_threshold, np.expand_dims(mean, axis=0), axis=0)
                gt_below_threshold = np.append(gt_below_threshold, np.expand_dims(y_bag, axis=0), axis=0)

    accuracy = correct_preds/n
    correct_pred_std = np.mean(correct_stds)
    wrong_pred_std = np.mean(wrong_stds)

    metrics, conf_mat = calc_wsi_prostate_cancer_metrics(gt_below_threshold, mean_below_threshold)
    save_metrics_and_conf_matrics(metrics, conf_mat, out_dir=os.path.join(out_dir, 'below_threshold/'))
    plt.figure()
    font = {'size': 14}
    plt.rc('font', **font)
    plt.hist(correct_stds, bins=20, range=(0.0, 0.1), facecolor='g', alpha=0.5, label='Correctly Classified Bags')
    plt.hist(wrong_stds, bins=20, range=(0.0, 0.1), facecolor='r', alpha=0.5, label='Incorrectly Classified Bags')
    plt.xlabel("Standard Deviation", size=16)
    plt.ylabel("Number of Bags", size=14)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(out_dir, 'std_distribution.png'))

    unc_metrics = {'bag_accuracy': accuracy, 'correct_pred_std': correct_pred_std, 'wrong_pred_std': wrong_pred_std}
    f = open(os.path.join(out_dir, 'uncertainty_metric.txt'), "w")
    f.write(str(unc_metrics))
    f.close()

