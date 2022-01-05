import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_metrics_and_conf_matrics(metrics, conf_matrix, out_dir='./out', grading='isup'):
    if grading == 'isup':
        classes = ['NC', 'G1', 'G2', 'G3', 'G4', 'G5']
    else:
        classes = ['NC', 'GS6', 'GS7', 'GS8', 'GS9', 'GS10']
    print(metrics)
    print(conf_matrix)
    print('Save to: ' + out_dir)

    f = open(os.path.join(out_dir, 'metrics.txt'), "w")
    f.write(str(metrics))
    f.close()

    print(str(conf_matrix))
    plt.figure()
    font = {'size': 16}
    plt.rc('font', **font)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels = classes)
    disp.plot(cmap='Blues', colorbar=False)

    conf_mat_path = os.path.join(out_dir, 'conf_matrix.png')
    # plt.show()
    plt.savefig(conf_mat_path)

    f = open(os.path.join(out_dir, 'conf_matrix.txt'), "w")
    f.write(str(conf_matrix))
    f.close()

