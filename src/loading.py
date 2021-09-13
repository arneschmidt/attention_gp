from typing import Dict
import pandas as pd
import numpy as np


def load_dataframe(df: pd.DataFrame, config: Dict):
    train_with_instance_labels = False # only use if some day we want to train with instance labels
    # we try to automatically derive the column names
    col_feature_prefix = config['col_feature_prefix']
    col_bag_label = config['col_bag_label']
    col_bag_name = config['col_bag_name']
    col_instance_label = config['col_instance_label']

    # find all feature columns
    col_features = []
    for col in df.columns:
        if col_feature_prefix in col:
            col_features.append(col)

    if col_bag_name in df.columns:
        bag_names_per_instance = df[col_bag_name].to_numpy().astype('str')
    else:
        bag_names_per_instance = np.array([])

    features = df[col_features].to_numpy().astype('float32')

    pi = None
    mask = None
    Z = None
    if col_bag_label in df.columns:
        bag_labels_per_instance = df[col_bag_label].to_numpy().astype('int')
    else:
        bag_labels_per_instance = np.array([])

    if col_instance_label in df.columns:
        instance_labels = (df[col_instance_label].to_numpy().astype("int"))  # instance_label column
        if train_with_instance_labels:
            pi = np.random.uniform(0, 0.1, size=len(df))  # -1 for untagged
            pi = np.where((0 == instance_labels), 0, pi)
            pi = np.where((0 < instance_labels), 1, pi)

            mask = np.where(instance_labels > -1, False, True)
    else:
        instance_labels = np.array([])

    return features, bag_labels_per_instance, bag_names_per_instance, instance_labels


def load_cnn_predictions(test_df, config):
    col_cnn_prediction = config['col_cnn_prediction']
    col_bag_cnn_prediction = config['col_bag_cnn_prediction']
    col_bag_cnn_probability = config['col_bag_cnn_probability']

    if col_cnn_prediction in test_df.columns:
        cnn_prediction = test_df[col_cnn_prediction].to_numpy().astype("float32")
    else:
        cnn_prediction = np.array([])
    if col_bag_cnn_prediction in test_df.columns:
        bag_cnn_prediction = test_df[col_bag_cnn_prediction].to_numpy().astype("float32")
    else:
        bag_cnn_prediction = np.array([])
    if col_bag_cnn_probability in test_df.columns:
        bag_cnn_probability = test_df[col_bag_cnn_probability].to_numpy().astype("float32")
    else:
        bag_cnn_probability = np.array([])

    return cnn_prediction, bag_cnn_prediction, bag_cnn_probability

def get_bag_level_information(features: np.array, bag_labels_per_instance: np.array, bag_names_per_instance: np.array,
                              pooling: str = 'avg'):
    """
    pooling: 'avg' or 'max'
    """
    bag_names = np.unique(bag_names_per_instance)
    bag_features = []
    bag_labels = []

    for bag_name in bag_names:
        bag_indices = (bag_names_per_instance == bag_name)
        inst_features_of_bag = features[bag_indices]

        if pooling == 'avg':
            bag_features_of_bag = np.mean(inst_features_of_bag, axis=0)
        else:
            vector_norm = np.linalg.norm(inst_features_of_bag, axis=0)
            argmax = np.argmax(vector_norm)
            bag_features_of_bag = inst_features_of_bag[argmax]
        bag_features.append(bag_features_of_bag)

        bag_gt_label = np.unique(bag_labels_per_instance[bag_indices])
        assert len(bag_gt_label) == 1  # make sure all bag labels are the same for one bag
        bag_labels.append(bag_gt_label)

    bag_features = np.array(bag_features)
    bag_labels = np.array(bag_labels)

    return bag_features, bag_labels, bag_names

