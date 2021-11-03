import os
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.utils import class_weight
from model_architecture import build_model
from evaluate import bag_level_evaluation
from utils.wsi_cancer_binary_utils import calc_wsi_cancer_binary_metrics
from utils.wsi_prostate_cancer_utils import calc_wsi_prostate_cancer_metrics
from utils.saving import save_metrics_and_conf_matrics
from sklearn.metrics import cohen_kappa_score
from mlflow_log import MLFlowCallback


class Model:
    """
    Class that contains the classification model. Wraps keras model.
    """
    def __init__(self, config: Dict, data_dims: int, n_training_points: int):
        self.n_training_points = n_training_points
        self.config = config
        model, instance_model, bag_level_uncertainty_model = build_model(config, data_dims, n_training_points)
        print(model.summary())
        self.model = model
        self.instance_model = instance_model
        self.bag_level_uncertainty_model = bag_level_uncertainty_model

    def load(self):
        self.model.load_weights(os.path.join(self.config['output_dir'], "models/model.h5"))

    def train(self, train_gen, val_gen):
        def scheduler(epoch, lr):
            if epoch < self.config["model"]["lr_decay_after_epoch"]:
                return lr
            else:
                return lr * tf.math.exp(self.config["model"]["lr_decay_factor"])

        callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

        callback_mlflow = MLFlowCallback(self.config)
        if self.config["model"]["class_weighted_loss"]:
            class_weights = self._calculate_class_weights(train_gen)
        else:
            class_weights = None
        self.model.fit(train_gen, epochs=int(self.config['model']['epochs']), validation_data=val_gen,
                       callbacks=[callback_lr, callback_mlflow], class_weight=class_weights)
        if self.config['model']['metrics_for_model_saving'] != 'None':
            self.model.set_weights(callback_mlflow.best_weights)
        if self.config['model']['save_model']:
            callback_mlflow._save_model()


    def test(self, test_gen):
        predictions = self.model.predict(test_gen)
        predictions = np.reshape(predictions, [-1, test_gen.labels[0].shape[0]])
        gt = test_gen.labels
        if self.config['data']['type'] == 'binary':
            metrics, conf_matrix = calc_wsi_cancer_binary_metrics(predictions, gt)
            # bag_level_evaluation(test_gen, self.bag_level_uncertainty_model)
        else:
            # predictions = np.reshape(predictions, [-1, test_gen.labels[0].shape[0]])
            metrics, conf_matrix = calc_wsi_prostate_cancer_metrics(gt, predictions)
        if self.bag_level_uncertainty_model is not None:
            uncertainty_metric = bag_level_evaluation(test_gen, self.bag_level_uncertainty_model)
        else:
            uncertainty_metric = {}
        self._output_saving(predictions, gt, metrics, conf_matrix, uncertainty_metric)

        return metrics, conf_matrix

    def _calculate_class_weights(self, train_gen):
        """
        Calculate class weights based on gt, pseudo and soft labels.
        :param training_targets: gt, pseudo and soft labels (fused)
        :return: class weight dict
        """
        labels = np.argmax(train_gen.labels, axis=1)
        classes = np.arange(0, self.config['data']['num_classes'])
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels)
        class_weights = {}
        for class_id in classes:
            class_weights[class_id] = class_weights_array[class_id]
        return class_weights

    def _output_saving(self, predictions, gt, metrics, conf_matrix, uncertainty_metric):

        if self.config['logging']['save_predictions']:
            with open(os.path.join(self.config['output_dir'], 'predictions.npy'), 'wb') as f:
                np.save(f, predictions)
            with open(os.path.join(self.config['output_dir'], 'gt.npy'), 'wb') as f:
                np.save(f, gt)
            # with open(os.path.join(self.config['output_dir'], 'metrics.txt'), 'wb') as f:
            #     f.write(str(metrics))
        save_metrics_and_conf_matrics(metrics= metrics, conf_matrix=conf_matrix, unc_metrics=uncertainty_metric,
                                      out_dir=self.config['output_dir'], grading=self.config['data']['type'])

        with open(os.path.join(self.config['output_dir'], 'model_summary.txt'), 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))


    def instance_pred_out(self, test_gen, test_instance_labels, test_bag_names_per_instance, test_instance_names):

        for i in range(len(test_gen)):
            x, y = test_gen[i]
            pred = self.instance_model.predict(np.expand_dims(x, axis=0))
            mean = np.mean(pred, axis=0)
            std = np.std(pred, axis=0)

            if i == 0:
                mean_concat = mean
                std_concat = std
            else:
                mean_concat = np.concatenate([mean_concat, mean], axis=0)
                std_concat = np.concatenate([std_concat, std], axis=0)

        df = pd.DataFrame()
        df['bag_name'] = test_bag_names_per_instance
        df['instance_labels'] = test_instance_labels
        df['instance_names'] = test_instance_names
        df['mean'] = mean_concat
        df['std'] = std_concat
        df.to_csv(os.path.join(self.config['output_dir'], 'instance_preds.csv'))


