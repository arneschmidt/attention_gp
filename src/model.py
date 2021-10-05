import tensorflow as tf
import numpy as np
from typing import Dict, Optional, Tuple
from model_architecture import build_model
from evaluate import bag_level_evaluation
from utils.wsi_cancer_binary_utils import calc_wsi_cancer_binary_metrics
from utils.wsi_prostate_cancer_utils import calc_wsi_prostate_cancer_metrics
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
        self.model = model
        self.instance_model = instance_model
        self.bag_level_uncertainty_model = bag_level_uncertainty_model


    def train(self, train_gen, val_gen):
        callback = MLFlowCallback(self.config)
        self.model.fit(train_gen, epochs=int(self.config['model']['epochs']), validation_data=val_gen,
                       callbacks=[callback])
        self.model.set_weights(callback.best_weights)

    def test(self, test_gen):
        predictions = self.model.predict(test_gen)
        gt = test_gen.labels
        if self.config['data']['type'] == 'binary':
            metrics = calc_wsi_cancer_binary_metrics(predictions, gt)
            conf_matrix = []
            # bag_level_evaluation(test_gen, self.bag_level_uncertainty_model)
        else:
            predictions = self.model.predict(test_gen)
            predictions = np.reshape(predictions, [-1, test_gen.labels[0].shape[0]])
            metrics, conf_matrix = calc_wsi_prostate_cancer_metrics(test_gen.labels, predictions)
        print(metrics)
        print(conf_matrix)
        return metrics, conf_matrix
