import os
import pandas as pd
import tensorflow as tf
import numpy as np

from loading import get_bag_level_information, load_dataframe
from data_generator import DataGenerator
from utils.wsi_prostate_cancer_utils import get_gleason_score_and_isup_grade


class Data():
    def __init__(self, config):
        self.config = config
        self.wsi_df = pd.read_csv(os.path.join(self.config['input_path'], self.config['wsi_file']))

    def generate_data(self, split: str):
        bag_names, bag_labels, features, bag_labels_per_instance, bag_names_per_instance, instance_labels = self.load(split)
        images, labels = self.prepare_bags(features, bag_names, bag_labels, bag_names_per_instance)

        if split == 'train':
            shuffle = True
        else:
            shuffle = False

        data_gen = DataGenerator(images, labels, shuffle)
        return data_gen

    def load(self, split: str):
        df = None
        if split =='train':
            df = pd.read_csv(os.path.join(self.config['input_path'], self.config['train_file']))
        elif split =='val':
            df = pd.read_csv(os.path.join(self.config['input_path'], self.config['val_file']))
        elif split =='test':
            df = pd.read_csv(os.path.join(self.config['input_path'], self.config['test_file']))

        features, bag_labels_per_instance, bag_names_per_instance, instance_labels = load_dataframe(df, self.config)
        if self.config['type'] == 'binary':
            indices = self.wsi_df['slide'].isin(np.unique(bag_names_per_instance))
            bag_names = np.array(self.wsi_df['slide'].loc[indices])
            bag_labels = tf.keras.utils.to_categorical(np.array(self.wsi_df['P'].loc[indices]))
        else:
            # TODO: find only wsi of the train/val/test split that is generated
            bag_names = np.unique(bag_names_per_instance)
            if 'isup_grade' not in self.wsi_df.columns:
                self.wsi_df = get_gleason_score_and_isup_grade(self.wsi_df)
            indices = self.wsi_df['slide_id'].isin(bag_names)
            if self.config['type'] == 'isup':
                bag_labels = tf.keras.utils.to_categorical(np.array(self.wsi_df['isup_grade'][indices]))
            elif self.config['type'] == 'gleason_score':
                bag_labels = tf.keras.utils.to_categorical(np.array(self.wsi_df['gleason_score'][indices]), num_classes=6)
            else:
                raise Exception('Choose valid dataset type (data: dataset_type:')
            # bag_labels = np.array(self.wsi_df['isup_grade'][indices])

        return bag_names, bag_labels, features, bag_labels_per_instance, bag_names_per_instance, instance_labels

    def prepare_bags(self, features, bag_names, bag_labels, bag_names_per_instance):
        bag_names = bag_names
        images = []
        labels = []

        for i in range(len(bag_names)):
            bag_name = bag_names[i]
            id_bool = (bag_name == bag_names_per_instance)
            bag_features = features[id_bool]
            if self.config['cut_bags']:
                bag_features = self.cut_bags(bag_features)

            images.append(bag_features)
            labels.append(bag_labels[i])

        return images, labels

    def cut_bags(self, features):
        max_n = 500
        norms = np.linalg.norm(features, axis=-1)
        sorted = np.argsort(norms)[::-1]
        selected_ids = sorted[0:max_n]
        features = features[selected_ids]

        return features



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


