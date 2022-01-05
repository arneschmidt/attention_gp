import pandas as pd
from utils.saving import save_metrics_and_conf_matrics

input_csv = 'dataset_dependent/sicapv2/arvaniti/conf_mat.csv'
output_dir = 'dataset_dependent/sicapv2/arvaniti/'
grading = 'GS'
# input_csv = 'dataset_dependent/panda/experiments/initial_experiments/silva/conf_mat.csv'
# output_dir = 'dataset_dependent/panda/experiments/initial_experiments/silva/'
# grading = 'isup'

df = pd.read_csv(input_csv, header=None)
mat = df.to_numpy()
save_metrics_and_conf_matrics({}, mat, output_dir, grading)