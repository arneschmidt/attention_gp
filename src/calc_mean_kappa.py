import numpy as np
import pandas as pd
models = ['baseline', 'att', 'att_gated', 'agp']

# PANDA eval
# files = [
#     [
#         'dataset_dependent/sicapv2/baseline/total/metrics.txt',
#         'dataset_dependent/sicapv2/baseline2/total/metrics.txt',
#         'dataset_dependent/sicapv2/baseline3/total/metrics.txt',
#         'dataset_dependent/sicapv2/baseline4/total/metrics.txt',
#     ],
#     [
#         'dataset_dependent/sicapv2/cnn/total/metrics.txt',
#         'dataset_dependent/sicapv2/cnn2/total/metrics.txt',
#         'dataset_dependent/sicapv2/cnn3/total/metrics.txt',
#         'dataset_dependent/sicapv2/cnn4/total/metrics.txt',
#     ],
#     [
#         'dataset_dependent/sicapv2/cnn_gated/total/metrics.txt',
#         'dataset_dependent/sicapv2/cnn_gated2/total/metrics.txt',
#         'dataset_dependent/sicapv2/cnn_gated3/total/metrics.txt',
#         'dataset_dependent/sicapv2/cnn_gated4/total/metrics.txt',
#     ],
#     [
#         'dataset_dependent/sicapv2/gp/total/metrics.txt',
#         'dataset_dependent/sicapv2/gp2/total/metrics.txt',
#         'dataset_dependent/sicapv2/gp3/total/metrics.txt',
#         'dataset_dependent/sicapv2/gp4/total/metrics.txt',
#     ]
# ]

# SICAPv2 eval
# files = [
#     [
#         'dataset_dependent/panda/experiments/initial_experiments/baseline/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/baseline2/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/baseline3/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/baseline4/metrics.txt',
#
#     ],
#     [
#         'dataset_dependent/panda/experiments/initial_experiments/cnn/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/cnn2/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/cnn3/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/cnn4/metrics.txt'
#     ],
#     [
#         'dataset_dependent/panda/experiments/initial_experiments/cnn_gated/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/cnn_gated2/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/cnn_gated3/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/cnn_gated4/metrics.txt'
#     ],
#     [
#         'dataset_dependent/panda/experiments/initial_experiments/default/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/default2/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/default3/metrics.txt',
#         'dataset_dependent/panda/experiments/initial_experiments/default4/metrics.txt'
#     ]
# ]

# Panda train, SICAP eval
files = [
    [
        'dataset_dependent/sicapv2/baseline/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/baseline2/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/baseline3/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/baseline4/train_panda_test_sicap/metrics.txt'
    ],
    [
        'dataset_dependent/sicapv2/cnn/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/cnn2/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/cnn3/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/cnn4/train_panda_test_sicap/metrics.txt'
    ],
    [
        'dataset_dependent/sicapv2/cnn_gated/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/cnn_gated2/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/cnn_gated3/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/cnn_gated4/train_panda_test_sicap/metrics.txt'
    ],
    [
        'dataset_dependent/sicapv2/gp/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/gp2/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/gp3/train_panda_test_sicap/metrics.txt',
        'dataset_dependent/sicapv2/gp4/train_panda_test_sicap/metrics.txt'
    ]
]

out = 'dataset_dependent/sicapv2/total_results_trained_on_panda.csv'

df = pd.DataFrame(columns=['model', 'mean_kappa', 'std_kappa', 'se_kappa'])

for i in range(len(models)):
    kappas = []
    for j in range(len(files[i])):
        with open(files[i][j]) as f:
            lines = f.readlines()
        kappa = float(str(lines)[34:40])
        kappas.append(kappa)
    kappas = np.array(kappas)
    mean_kappa = np.mean(kappas)
    std_kappa = np.std(kappas)
    se_kappa = std_kappa/np.sqrt(np.size(kappas))
    df_row = {'model': models[i], 'mean_kappa': mean_kappa, 'std_kappa': std_kappa, 'se_kappa': se_kappa}
    df = df.append(df_row, ignore_index=True)

df.to_csv(out)
