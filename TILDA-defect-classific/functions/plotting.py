import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_curve, auc, RocCurveDisplay


def plot_ROC(y_actual, y_pred, estimator_name='2-class CNN model'):
    fpr, tpr, thresholds = roc_curve(y_actual,y_pred)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, 
                                      estimator_name=estimator_name)
    display.plot()
    plt.title("Receiver Operating Characteristics (ROC)")
    plt.grid(ls = ':')
    plt.show()


def plot_2class_histograms(y_actual, y_predict, label_dict, stepsize = 0.01, density = False):
    y_prob = y_predict[:,1]
    y_actual_labels =  np.array([y for y in map(lambda number: label_dict[number], list(y_actual))])
    unique_labels = list(set(y_actual_labels))
    bins = np.arange(0,1+stepsize, stepsize)

    plt.title("Distributions of Predicted Probabilities")
    for unique_label in unique_labels:
        plot_label_samples = y_prob[y_actual_labels == unique_label]
        plt.hist(plot_label_samples, label = unique_label, bins = bins, alpha = 0.5, density = density)
    plt.xlabel("y_predict (probability)")
    plt.ylabel("frequency")
    plt.xticks(np.arange(0,1.1,0.1))
    plt.xlim(-0.01,1.01)
    plt.grid(ls = ':')
    plt.legend()
    plt.show()