import pandas as pd
import tensorflow as tf
import numpy as np

from loading import get_bag_level_information, load_dataframe
from data_generator import DataGenerator


class Data():
    def __init__(self, config):
        self.config = config
        self.wsi_df = pd.read_csv(self.config['path_wsi_df'])

    def generate_data(self, split: str):
        bag_names, bag_labels, features, bag_labels_per_instance, bag_names_per_instance, instance_labels = self.load(split)
        images, labels = self.prepare_bags(features, bag_names_per_instance, bag_labels_per_instance)

        if split == 'train':
            shuffle = True
        else:
            shuffle = False

        data_gen = DataGenerator(images, labels, shuffle)
        return data_gen

    def load(self, split: str):
        df = None
        if split =='train':
            df = pd.read_csv(self.config['path_train_df'])
        elif split =='val':
            df = pd.read_csv(self.config['path_val_df'])
        elif split =='test':
            df = pd.read_csv(self.config['path_test_df'])

        features, bag_labels_per_instance, bag_names_per_instance, instance_labels = load_dataframe(df, self.config)
        indices = self.wsi_df['slide'].isin(np.unique(bag_names_per_instance))
        bag_names = self.wsi_df['slide'].loc[indices]
        bag_labels = self.wsi_df['P'].loc[indices]

        return bag_names, bag_labels, features, bag_labels_per_instance, bag_names_per_instance, instance_labels

    def prepare_bags(self, features, bag_names_per_instance, bag_labels_per_instance):
        bag_names = np.unique(bag_names_per_instance)
        images = []
        labels = []

        for i in range(len(bag_names)):
            bag_name = bag_names[i]
            id_bool = (bag_name == bag_names_per_instance)

            images.append(features[id_bool])
            labels.append(np.unique(bag_labels_per_instance[id_bool]))

        return images, labels


    # def _prepare_data(self, ds, features, bag_names_per_instance):
    #     def _fill_the_bags(x, y):
    #         bag_feat = features[bag_names_per_instance==x]
    #         x = bag_feat
    #         return x, y
    #
    #     ds = ds.cache()
    #     # ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    #     ds = ds.batch(1)
    #     ds = ds.map(lambda x, y: _fill_the_bags(x, y))
    #     # ds = ds.map(lambda x, y: _reshape(x, y))
    #     ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    #     return ds
    #
    # def fill_the_bags(self, data_generator, features, bag_names_per_instance):
    #     for x, y in data_generator:
    #         bag_feat = tf.convert_to_tensor(features[bag_names_per_instance == x], dtype=tf.float32)
    #         x = bag_feat
    #         yield x, y


