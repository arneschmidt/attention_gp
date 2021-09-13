import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import yaml
import argparse
import tensorflow as tf
import pandas as pd

from model import build_model
from metrics import Metrics
from data import Data
from evaluate import bag_level_evaluation, att_evaluation


def main():
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0,1,2,3,4], False)

    save_dir = 'out/'
    os.makedirs(save_dir, exist_ok=True)

    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--config", "-c", type=str, default="./config.yaml",
                        help="Config path (yaml file expected) to default config.")
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.full_load(file)


    data = Data(config)
    train_gen = data.generate_data('train')
    val_gen = data.generate_data('val')


    model, instance_model, bag_level_uncertainty_model = build_model()
    model.fit(train_gen, epochs=2, validation_data=val_gen)
    test_gen = data.generate_data('test')
    model.evaluate(test_gen)

    bag_level_evaluation(test_gen, bag_level_uncertainty_model)
    # att_evaluation(instance_model, test_gen)

if __name__ == '__main__':
    main()