import os
import argparse
import numpy as np
from utils.wsi_prostate_cancer_utils import calc_wsi_prostate_cancer_metrics

parser = argparse.ArgumentParser(description="Cancer Classification")
parser.add_argument("--input_dir", "-i", type=str, default="./config.yaml",
                    help="Config path (yaml file expected) to default config.")
args = parser.parse_args()


input_dir = args.input_dir
out_dir = input_dir + "total"

input_paths = [input_dir + 'split1', input_dir + 'split2', input_dir + 'split3', input_dir + 'split4']
total_predictions = []
total_gt = []
for i in range(len(input_paths)):
    # with open(os.path.join(input_paths[i], 'predictions.npy'), 'r') as f:
    predictions = np.load(os.path.join(input_paths[i], 'predictions.npy'))
    # with open(os.path.join(input_paths[i], 'gt.npy'), 'r') as f:
    gt = np.load(os.path.join(input_paths[i], 'gt.npy'))

    if i==0:
        total_predictions = predictions
        total_gt = gt
    else:
        total_predictions = np.concatenate([total_predictions, predictions], axis=0)
        total_gt = np.concatenate([total_gt, gt], axis=0)

metrics, conf_matrix = calc_wsi_prostate_cancer_metrics(total_gt, total_predictions)

print(metrics)
print(conf_matrix)

f = open(os.path.join(out_dir, 'metrics.txt'), "w")
f.write(str(metrics))
f.close()
f = open(os.path.join(out_dir, 'conf_matrix.txt'), "w")
f.write(str(conf_matrix))
f.close()


