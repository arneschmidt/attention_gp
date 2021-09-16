import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

def bag_level_evaluation(test_gen, bag_level_uncertainty_model: tf.keras.Model):
    # test_data = test_gen.as_numpy_iterator()

    n = len(test_gen)
    correct_preds = 0
    correct_stds = 0
    wrong_preds = 0
    wrong_stds = 0
    preds = bag_level_uncertainty_model.predict(test_gen)

    for i in range(n):
        y_bag = test_gen.labels[i]
        bag_pred = preds[i]
        mean = np.mean(bag_pred, axis=0)[1]
        std = np.std(bag_pred, axis=0)[1]
        correct_pred = (np.round(mean) == y_bag)[0]
        if correct_pred:
            correct_preds += 1
            correct_stds += std
        else:
            wrong_preds += 1
            wrong_stds += std

    accuracy = correct_preds/n
    correct_pred_std = correct_stds/correct_preds
    wrong_pred_std = wrong_stds/wrong_preds

    print('Bag Accuracy: ' + str(accuracy)
          + '; correct_pred_std: ' + str(correct_pred_std)
          + '; wrong_pred_std: ' + str(wrong_pred_std))

