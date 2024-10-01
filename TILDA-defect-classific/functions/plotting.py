import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_curve, auc, RocCurveDisplay


def plot_ROC(y_actual, y_pred, filename = None, estimator_name='2-class CNN model'):
    fpr, tpr, thresholds = roc_curve(y_actual,y_pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, 
                                      estimator_name=estimator_name)
    display.plot()
    plt.title("Receiver Operating Characteristics (ROC)")
    plt.grid(ls = ':')
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_2class_histograms(y_actual, y_predict, label_dict, stepsize = 0.01, density = False, semilog = False, holdout = False):
    y_prob = y_predict[:,1]
    y_actual_labels =  np.array([y for y in map(lambda number: label_dict[number], list(y_actual))])
    unique_labels = list(set(y_actual_labels))
    bins = np.arange(0,1+stepsize, stepsize)
    if holdout:
        plt.title("Distributions of Predicted Probabilities (holdout)")
    else: 
        plt.title("Distributions of Predicted Probabilities")
    for unique_label in unique_labels:
        plot_label_samples = y_prob[y_actual_labels == unique_label]
        plt.hist(plot_label_samples, label = unique_label, bins = bins, alpha = 0.5, density = density)
    plt.xlabel("y_predict (probability)")
    plt.ylabel("Probability Density (%)")
    if semilog:
        plt.semilogy()
    plt.xticks(np.arange(0,1.1,0.1))
    plt.xlim(-0.01,1.01)
    plt.grid(ls = ':')
    plt.legend()
    plt.show()


def plot_CNN_filters(image_learner, caption, layer_num, path = "../notebook_CNN/plots/",
                     filename = '2_class_4L_16x32.png', dpi = 'figure', size = 100, max_num_rows = 32):
    conv2d_16_layer = image_learner.layers[layer_num]
    vmin = conv2d_16_layer.kernel.numpy().min()
    vmax = conv2d_16_layer.kernel.numpy().max()
    
    d0 = conv2d_16_layer.kernel.shape[2]
    d1 = conv2d_16_layer.kernel.shape[3]
    print(d0, d1, int(d0*d1/8))
    if d1 <= 8:
        fig, ax = plt.subplots(1, 8, figsize=(100,100), layout="compressed")
        for col in range(d1):
            filters_0 = conv2d_16_layer.kernel[:,:,0,col]
            # print(f'plot positions: col: {col}, index: {col}')
            ax[col].imshow(filters_0, cmap = 'gray')
            ax[col].set_xticklabels([], fontsize = 0)
            ax[col].set_yticklabels([], fontsize = 0)
            ax[col].xaxis.set_ticks_position('none') 
            ax[col].yaxis.set_ticks_position('none') 
            ax[col].text(0, 1, caption, size= size, color = 'yellow', weight='bold')
        # print()
    else:
        num_rows = int(d0*d1/8)
        if num_rows > max_num_rows:
            num_rows = max_num_rows
        fig, ax = plt.subplots(num_rows, 8, figsize=(100,100), layout="compressed")
        for indx0 in range(d0):
            for indx1 in range(d1):
                filters_0 = conv2d_16_layer.kernel[:,:,indx0,indx1]
                col = indx1%8
                row = int((indx0*2*d0+indx1)/8)
                if row + 1 <= max_num_rows:
                    # print(f'plot positions: row: {row}, col: {col}, layer indices: {indx0, indx1}')
                    ax[row, col].imshow(filters_0, cmap = 'gray', vmin = vmin, vmax = vmax)
                    ax[row, col].set_xticklabels([], fontsize = 0)
                    ax[row, col].set_yticklabels([], fontsize = 0)
                    ax[row, col].xaxis.set_ticks_position('none') 
                    ax[row, col].yaxis.set_ticks_position('none') 
                    ax[row, col].text(0, 1, caption, size= size, color = 'yellow', weight='bold')
            # print()
    if filename is not None:
        image = plt.savefig(path + filename, bbox_inches='tight',dpi = dpi)
    plt.show()


def plot_confusion_matrix(y_actual, y_pred, normalize = 'pred'):

    labels = sorted(list(set(y_actual)))
    
    if normalize == 'true':
        plt.title("Confusion Matrix (Recall)")
        cmap = 'Greens'
    elif normalize == 'pred':
        plt.title("Confusion Matrix (Precision)")
        cmap = 'Blues'
    elif normalize == 'all':
        plt.title("Confusion Matrix (Accuracy)")
        cmap = 'Purples'
    elif normalize is None:
        plt.title("Confusion Matrix (Counts)")
        cmap = 'Reds'
    else:
        return
        
    print('')
    cm = confusion_matrix(y_true=y_actual, y_pred=y_pred, normalize=normalize)
    print(cm)
    num_of_diagonal = np.sum([cm[x,x] for x in range(cm.shape[0])])
    print(f"sum of diagonal = {num_of_diagonal}")
    mc_df = pd.DataFrame(cm,
                         index=labels, 
                         columns=labels
                        )
    sns.heatmap(mc_df, annot =True, 
                # fmt="d",
                color = 'r',
                cmap=plt.get_cmap(cmap))
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.xlabel('y_pred')
    plt.ylabel('y_actual')
    plt.show()
    return