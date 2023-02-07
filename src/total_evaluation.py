import os
import argparse
import numpy as np
from utils.wsi_prostate_cancer_utils import calc_wsi_prostate_cancer_metrics
from utils.saving import save_metrics_and_conf_matrics

# parser = argparse.ArgumentParser(description="Cancer Classification")
# parser.add_argument("--input_dir", "-i", type=str, default="./config.yaml",
#                     help="Config path (yaml file expected) to default config.")
# args = parser.parse_args()
#
#
# input_dir = args.input_dir
input_folder = 'dataset_dependent/sicapv2/ablation_studies/'
models = ['gp_f_32', 'gp_f_128', 'gp_f_256', 'gp_ip_16', 'gp_ip_32', 'gp_ip_128', 'gp_relu', 'gp_tanh'] # 'gp_ip_256',
runs = ['', '_1', '_2', '_3'] #, '_4']
result_mean = {}
result_se = {}
for model in models:
    model_means = []

    for run in runs:
        input_dir = input_folder + model + run + '/'
        out_dir = input_dir + "total"

        input_paths = [input_dir + 'split1', input_dir + 'split2', input_dir + 'split3', input_dir + 'split4']
        total_predictions = []
        total_gt = []

        cohens_kappas = []

        for i in range(len(input_paths)):

            pred_path = os.path.join(input_paths[i], 'predictions.npy')
            if os.path.exists(pred_path):
                predictions = np.load(pred_path)
                gt = np.load(os.path.join(input_paths[i], 'gt.npy'))
            else:
                print('Predictions not available for ' + input_dir)
                break

            metrics, conf_matrix = calc_wsi_prostate_cancer_metrics(gt, predictions)
            cohens_kappas.append(metrics['test_cohens_quadratic_kappa'])
            if i==0:
                total_predictions = predictions
                total_gt = gt
            else:
                total_predictions = np.concatenate([total_predictions, predictions], axis=0)
                total_gt = np.concatenate([total_gt, gt], axis=0)

        metrics, conf_matrix = calc_wsi_prostate_cancer_metrics(total_gt, total_predictions)

        mean_cohens_kappa = np.mean(np.array(cohens_kappas))
        std_cohens_kappa = np.std(np.array(cohens_kappas))
        metrics['mean_cohens_kappa'] = mean_cohens_kappa
        metrics['std_cohens_kappa'] = std_cohens_kappa

        # save_metrics_and_conf_matrics(metrics, conf_matrix, out_dir=out_dir, grading='GS')

        model_means.append(mean_cohens_kappa)

    final_mean = np.mean(model_means)
    std_kappa = np.std(model_means)
    final_se = std_kappa/np.sqrt(np.size(model_means))

    result_mean[model] = final_mean
    result_se[model] = final_se

print('Save to: ' + input_folder)

f = open(os.path.join(input_folder, 'metrics.txt'), "w")
f.write('Mean Kappa: ')
f.write(str(result_mean))
f.write('SE Kappa: ')
f.write(str(result_se))
f.close()



