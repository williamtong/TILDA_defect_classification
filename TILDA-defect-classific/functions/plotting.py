import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_curve, auc, RocCurveDisplay


def plot_ROC(y_actual, y_pred, filename=None, estimator_name='2-class CNN model'):
    '''
    Plots the Receiver Operating Characteristics (ROC) curve and prints the
    Area Under the Curve (ROC-AUC). A perfect classifier has AUC = 1.0;
    a random classifier has AUC = 0.5.

    Input:
    y_actual (array):        True binary labels (0 or 1).
    y_pred (array):          Predicted probabilities for the positive class.
    filename (str or None):  If provided, saves the plot to this path.
    estimator_name (str):    Label shown in the plot legend.
    '''
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                              estimator_name=estimator_name)
    display.plot()
    plt.title("Receiver Operating Characteristics (ROC)")
    plt.grid(ls=':')
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_2class_histograms(y_actual, y_predict, label_dict, stepsize=0.01,
                           density=False, semilog=False, holdout=False):
    '''
    Plots overlapping histograms of the model's predicted probabilities,
    one histogram per class. Useful for visualizing how well the model
    separates the two classes — a well-trained model will show two distinct,
    non-overlapping distributions.

    Input:
    y_actual (array):       True integer class labels.
    y_predict (array):      Model output probabilities, shape (N, 2). Column 1
                            is the predicted probability of the positive class.
    label_dict (dict):      Maps integer class codes to class name strings.
    stepsize (float):       Width of each histogram bin.
    density (bool):         If True, normalize histograms to probability density.
    semilog (bool):         If True, use a log scale on the y-axis.
    holdout (bool):         If True, appends "(holdout)" to the plot title.
    '''
    # Extract the predicted probability of the positive (defect) class.
    y_prob = y_predict[:,1]
    y_actual_labels = np.array([y for y in map(lambda number: label_dict[number], list(y_actual))])
    unique_labels = list(set(y_actual_labels))
    bins = np.arange(0, 1+stepsize, stepsize)

    if holdout:
        plt.title("Distributions of Predicted Probabilities (holdout)")
    else:
        plt.title("Distributions of Predicted Probabilities")

    # Plot one histogram per class, overlaid with 50% transparency.
    for unique_label in unique_labels:
        plot_label_samples = y_prob[y_actual_labels == unique_label]
        plt.hist(plot_label_samples, label=unique_label, bins=bins, alpha=0.5, density=density)

    plt.xlabel("y_predict (probability)")
    plt.ylabel("Probability Density (%)")
    if semilog:
        plt.semilogy()
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlim(-0.01, 1.01)
    plt.grid(ls=':')
    plt.legend()
    plt.show()


def plot_CNN_filters(image_learner, caption, layer_num, path="../notebook_CNN/plots/",
                     filename='2_class_4L_16x32.png', dpi='figure', size=100, max_num_rows=32):
    '''
    Visualizes the learned convolutional kernels (filters) of a specified layer.
    Each filter is displayed as a small grayscale image. Examining filters can
    give intuition about what low-level features (edges, textures) the model
    has learned to detect.

    Filters are arranged in a grid with 8 columns. If the layer has 8 or fewer
    filters, they are shown in a single row. Otherwise they wrap across multiple
    rows, capped at max_num_rows.

    Input:
    image_learner:          Trained Keras model.
    caption (str):          Text label overlaid on each filter image.
    layer_num (int):        Index of the convolutional layer to visualize.
    path (str):             Directory to save the output image.
    filename (str):         Filename for the saved image.
    dpi (str or int):       Resolution for the saved image.
    size (int):             Font size for the caption overlay.
    max_num_rows (int):     Maximum number of rows in the filter grid.
    '''
    conv2d_16_layer = image_learner.layers[layer_num]
    # Get the global min/max of the kernel weights for a consistent color scale across filters.
    vmin = conv2d_16_layer.kernel.numpy().min()
    vmax = conv2d_16_layer.kernel.numpy().max()

    # d0 = number of input channels, d1 = number of filters (output channels).
    d0 = conv2d_16_layer.kernel.shape[2]
    d1 = conv2d_16_layer.kernel.shape[3]
    print(d0, d1, int(d0*d1/8))

    if d1 <= 8:
        # Single row layout for layers with 8 or fewer filters.
        fig, ax = plt.subplots(1, 8, figsize=(100, 100), layout="compressed")
        for col in range(d1):
            filters_0 = conv2d_16_layer.kernel[:,:,0,col]
            ax[col].imshow(filters_0, cmap='gray')
            ax[col].set_xticklabels([], fontsize=0)
            ax[col].set_yticklabels([], fontsize=0)
            ax[col].xaxis.set_ticks_position('none')
            ax[col].yaxis.set_ticks_position('none')
            ax[col].text(0, 1, caption, size=size, color='yellow', weight='bold')
    else:
        # Multi-row layout: wrap every 8 filters onto a new row.
        num_rows = int(d0*d1/8)
        if num_rows > max_num_rows:
            num_rows = max_num_rows
        fig, ax = plt.subplots(num_rows, 8, figsize=(100, 100), layout="compressed")
        for indx0 in range(d0):
            for indx1 in range(d1):
                filters_0 = conv2d_16_layer.kernel[:,:,indx0,indx1]
                col = indx1 % 8
                row = int((indx0*2*d0+indx1)/8)
                if row + 1 <= max_num_rows:
                    # Use shared vmin/vmax so filter magnitudes are visually comparable.
                    ax[row, col].imshow(filters_0, cmap='gray', vmin=vmin, vmax=vmax)
                    ax[row, col].set_xticklabels([], fontsize=0)
                    ax[row, col].set_yticklabels([], fontsize=0)
                    ax[row, col].xaxis.set_ticks_position('none')
                    ax[row, col].yaxis.set_ticks_position('none')
                    ax[row, col].text(0, 1, caption, size=size, color='yellow', weight='bold')

    if filename is not None:
        plt.savefig(path + filename, bbox_inches='tight', dpi=dpi)
    plt.show()


def plot_confusion_matrix(y_actual, y_pred, normalize='pred'):
    '''
    Plots a color-coded confusion matrix heatmap. Supports four normalization modes,
    each emphasizing a different aspect of model performance:
      None:   Raw counts — how many samples fell into each cell.
      'all':  Normalized by total samples — each cell is a fraction of the whole dataset.
      'pred': Normalized by predicted class (columns) — equivalent to Precision per class.
      'true': Normalized by actual class (rows) — equivalent to Recall per class.

    The sum of the diagonal is printed to the console as a quick overall accuracy check.

    Input:
    y_actual (list):          True class name strings.
    y_pred (list):            Predicted class name strings.
    normalize (str or None):  Normalization mode: None, 'all', 'pred', or 'true'.
    '''
    labels = sorted(list(set(y_actual)))

    # Select title and color map based on normalization mode.
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
    # Sum of diagonal = overall accuracy (for 'all') or a weighted average (for other modes).
    num_of_diagonal = np.sum([cm[x,x] for x in range(cm.shape[0])])
    print(f"sum of diagonal = {num_of_diagonal}")

    mc_df = pd.DataFrame(cm, index=labels, columns=labels)
    sns.heatmap(mc_df, annot=True,
                color='r',
                cmap=plt.get_cmap(cmap))
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel('y_pred')
    plt.ylabel('y_actual')
    plt.show()
    return