# TODO
# perform ablation studies

import os
import collections
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import yaml
import argparse
import tensorflow as tf

from model import Model
from mlflow_log import MLFlowLogger
from data import Data


def main(config):
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    save_dir = config['output_dir']
    os.makedirs(save_dir, exist_ok=True)

    # load the data
    data = Data(config['data'])
    if not config['model']['load_model']:
        train_gen = data.generate_data('train')
        val_gen = data.generate_data('val')
        n_training_points = len(train_gen)
    else:
        n_training_points = 100 # arbitrary, not used
    test_gen = data.generate_data('test')

    # logger logs the parameters and metrics with mlflow
    logger = MLFlowLogger(config)
    logger.config_logging()

    # create the model
    model = Model(config, test_gen.images[0][0].shape, n_training_points)

    # train the model or load existing weights
    if not config['model']['load_model']:
        model.train(train_gen, val_gen)
    else:
        model.load()

    # test the model on the test set, log final metrics
    metrics, conf_matrices = model.test(test_gen)
    logger.test_logging(metrics)


def config_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = config_update(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def load_configs(args):
    with open(args.default_config) as file:
        config = yaml.full_load(file)
    with open(config["data"]["dataset_config"]) as file:
        config_data_dependent = yaml.full_load(file)

    config = config_update(config, config_data_dependent)

    if args.experiment_config != 'None':
        with open(args.experiment_config) as file:
            exp_config = yaml.full_load(file)
        config = config_update(config, exp_config)

    return config


if __name__ == "__main__":
    print('Load configuration')
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--default_config", "-dc", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    parser.add_argument("--experiment_config", "-ec", type=str, default="None",
                        help="Config path to experiment config. Parameters will override defaults. Optional.")
    args = parser.parse_args()
    config = load_configs(args)

    if config['logging']['run_name'] == 'auto':
        config['logging']['run_name'] = args.experiment_config.split('/')[-2]

    print('Create output folder')
    config['output_dir'] = os.path.join(config['data']['artifact_dir'], config['logging']['run_name'])
    os.makedirs(config['output_dir'], exist_ok=True)
    print('Output will be written to: ', config['output_dir'])

    main(config)