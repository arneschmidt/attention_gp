import pandas as pd
from utils.saving import save_metrics_and_conf_matrics

input_csv = 'dataset_dependent/panda/experiments/initial_experiments/silva/conf_mat.csv'
output_dir = 'dataset_dependent/panda/experiments/initial_experiments/silva/'
grading = 'GS'

df = pd.read_csv(input_csv, header=None)
mat = df.to_numpy()
save_metrics_and_conf_matrics({}, mat, None, output_dir, grading)