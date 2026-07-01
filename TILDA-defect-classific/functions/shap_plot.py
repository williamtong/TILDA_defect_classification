import numpy as np
import matplotlib.pyplot as plt
import shap

# Note: these imports use bare module names because the notebooks that call this file
# execute os.chdir() into the functions/ directory before importing, making this the
# working directory at runtime.
from models import get_class_weight, preprocessor, plot_confusion_matrix, compile_image_model, \
                    CNN_model, label_dict, revlabel_dict, recreate_labels, map_path_to_augmented

from preprocessing import upsample


# --- SHAP value sign filters ---
# SHAP values can be positive (evidence FOR a class) or negative (evidence AGAINST a class).
# These two helper functions zero out one sign so the two contributions can be visualized
# separately as overlays on the original image.

def using_where_neg(x):
    """Zero out negative values, retaining only positive SHAP contributions."""
    return np.where(x < 0, 0, x)
# Vectorized version for efficient element-wise application to numpy arrays.
v_pospass_filter = np.vectorize(using_where_neg)

def using_where_pos(x):
    """Zero out positive values, retaining only negative SHAP contributions."""
    return np.where(x > 0, 0, x)
# Vectorized version for efficient element-wise application to numpy arrays.
v_negpass_filter = np.vectorize(using_where_pos)


def plot_4class_shap_figures(shap_values, 
                             features_test_array, 
                             y_predict, 
                             image_num, 
                             label_dict, 
                             labels_test, size = 30, 
                             filename = None):
    '''
    Visualizes SHAP pixel-level explanations for a single image from the 4-class defect
    identification model (single kernel size). Produces two plots:
      1. The raw grayscale image with actual and predicted labels.
      2. A 4-row x 2-column grid of SHAP overlay images, one row per defect class.
         Left column (blue):  pixels that argue AGAINST that class.
         Right column (red):  pixels that argue FOR that class.
    The class with the most net red signal (relative to blue) wins the prediction.

    Input:
    shap_values (numpy array):       SHAP values from shap.DeepExplainer, shape
                                     (n_images, height, width, channels, n_classes).
    features_test_array (numpy array): Test images, shape (n_images, height, width, channels).
    y_predict (numpy array):         Model output probabilities, shape (n_images, n_classes).
    image_num (int):                 Index of the image to visualize.
    label_dict (dict):               Maps class integer codes to class name strings,
                                     e.g. {0: 'objects', 1: 'hole', 2: 'oil_spot', 3: 'thread_error'}.
    labels_test (list):              Actual integer class labels for the test set.
    size (int):                      Font size for on-image text annotations.
    filename (str or None):          If provided, saves both plots as .png files.
                                     The raw image is saved with a '_raw.png' suffix.
    '''
    print(f'image_num: {image_num}')

    # Convert raw model output probabilities to percentages rounded to 1 decimal place.
    y_prob = y_predict[image_num]
    y_prob = np.round((y_prob*10000).astype(int)/100, 1)
    print(f'Probabilities: {y_prob}')
    print(f'actual: {labels_test[image_num], label_dict[labels_test[image_num]]}, predicted{y_prob.argmax(), label_dict[y_prob.argmax()]}')

    # --- Plot 1: Raw grayscale image with actual and predicted class labels ---
    plt.figure(figsize = (7,7))
    plt.imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
               vmin = features_test_array[image_num,:,:,0].min(), 
               vmax = features_test_array[image_num,:,:,0].max(),
               alpha = 1)
    plt.text(0, 3, 
             f"actual: '{label_dict[labels_test[image_num]]}'", 
             size= int((size)/2), color = 'black', weight='bold')
    plt.text(0, 7, 
             f"predicted: '{label_dict[y_prob.argmax()]}'", 
             size= int((size)/2), color = 'black', weight='bold')
    plt.xticks([], fontsize = 0)
    plt.yticks([], fontsize = 0)
    if filename is not None:
        rawfilename = filename.replace(".png", "_raw.png")
        plt.savefig(rawfilename, bbox_inches='tight')
    plt.show()

    # --- Plot 2: SHAP overlay grid (4 classes x 2 columns: AGAINST / FOR) ---

    # Extract the SHAP values for this specific image.
    # Shape: (height, width, n_classes), one SHAP map per class.
    shap_values_image = shap_values[image_num,:,:,0,:]

    # Use the global absolute max as a shared color scale so all panels are comparable.
    vmax = np.max([-shap_values_image.min(), shap_values_image.max()])

    fig, ax = plt.subplots(nrows = 4, ncols=2, figsize=(50,50), layout="compressed")

    # Separate positive SHAP values (evidence FOR a class) from negative (evidence AGAINST).
    pos_shap_values_image = v_pospass_filter(shap_values_image)
    neg_shap_values_image = v_negpass_filter(shap_values_image)

    # One row per defect class; columns: [AGAINST (blue), FOR (red)].
    for l in range(4):

        # --- Right column (ax[l,1]): pixels supporting this class (positive SHAP, red) ---
        shap_image = pos_shap_values_image[:,:,l]
        ax[l,1].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,1].text(0, 3, f"shap pixels FOR '{label_dict[l]}' classification", size= size, color = 'brown', weight='bold')
        ax[l,1].text(0, 7, f"predicted probabilty = {y_prob[l]}%", size= size, color = 'brown', weight='bold')
        # Overlay positive SHAP values in red at 50% transparency.
        ax[l,1].imshow(shap_image, vmin = 0, vmax = vmax, cmap = 'Reds', alpha=0.5)
        ax[l,1].set_xticklabels([], fontsize = 0)
        ax[l,1].set_yticklabels([], fontsize = 0)
        ax[l,1].xaxis.set_ticks_position('none')

        # --- Left column (ax[l,0]): pixels arguing against this class (negative SHAP, blue) ---
        # Negate the (already-negative) values so the overlay magnitude is positive for imshow.
        shap_image = -neg_shap_values_image[:,:,l]
        ax[l,0].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,0].text(0, 3, f"shap pixels AGAINST '{label_dict[l]}' classification", size= size, color = 'mediumblue', weight='bold')
        # Overlay negative SHAP magnitudes in blue at 50% transparency.
        ax[l,0].imshow(shap_image, vmin = 0, vmax = vmax, cmap = 'Blues', alpha=0.5)
        ax[l,0].set_xticklabels([], fontsize = 0)
        ax[l,0].set_yticklabels([], fontsize = 0)
        ax[l,0].xaxis.set_ticks_position('none')

    if filename is not None:
        print(f'filename = {filename}')
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_2class_shap_figures(shap_values, 
                             features_test_array, 
                             y_predict, image_num, 
                             label_dict, 
                             labels_test, 
                             size = 30, 
                             filename = None):
    '''
    Visualizes SHAP pixel-level explanations for a single image from the 2-class defect
    detection model. Produces two plots:
      1. The raw grayscale image with actual and predicted labels.
      2. A side-by-side pair of SHAP overlay images:
         Left (blue):  pixels supporting the GOOD classification.
         Right (red):  pixels supporting the DEFECT classification.

    Input:
    shap_values (numpy array):       SHAP values from shap.DeepExplainer, shape
                                     (n_images, height, width, channels, n_classes).
    features_test_array (numpy array): Test images, shape (n_images, height, width, channels).
    y_predict (numpy array):         Model output probabilities, shape (n_images, n_classes).
    image_num (int):                 Index of the image to visualize.
    label_dict (dict):               Maps class integer codes to class name strings,
                                     e.g. {0: 'good', 1: 'defect'}.
    labels_test (list):              Actual integer class labels for the test set.
    size (int):                      Font size for on-image text annotations.
    filename (str or None):          If provided, saves both plots as .png files.
                                     The raw image is saved with a '_raw.png' suffix.
    '''
    print(f'image_num: {image_num}')

    # Convert raw model output probabilities to percentages rounded to 1 decimal place.
    y_prob = y_predict[image_num]
    y_prob = np.round((y_prob*10000).astype(int)/100, 1)
    print(f'Probabilities: {y_prob}')
    print(f'actual: {labels_test[image_num], label_dict[labels_test[image_num]]}, \
    predicted{y_predict[image_num].argmax(), label_dict[y_predict[image_num].argmax()]}')

    # Extract the SHAP values for this specific image.
    # Shape: (height, width, n_classes).
    shap_values_image = shap_values[image_num,:,:,0,:]

    full_image = features_test_array[image_num,:,:,0]

    # --- Plot 1: Raw grayscale image with actual and predicted class labels ---
    plt.figure(figsize = (7,7))
    plt.imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
               vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
               alpha = 1)
    plt.text(0, 3, 
             f"actual: '{label_dict[labels_test[image_num]]}'", 
             size= int((size)/4), color = 'black', weight='bold')
    plt.text(0, 7, 
             f"predicted: '{label_dict[y_prob.argmax()]}'", 
             size= int((size)/4), color = 'black', weight='bold')
    plt.xticks([], fontsize = 0)
    plt.yticks([], fontsize = 0)
    if filename is not None:
        rawfilename = filename.replace(".png", "_raw.png")
        plt.savefig(rawfilename, bbox_inches='tight')
    plt.show()

    # --- Plot 2: Side-by-side SHAP overlays for GOOD (left) and DEFECT (right) ---

    # Separate positive SHAP values (evidence FOR a class) from negative (evidence AGAINST).
    # For the 2-class case, class 0 = good, class 1 = defect.
    pos_shap_image = v_pospass_filter(shap_values_image)
    # Negate so the AGAINST magnitudes are positive for imshow.
    neg_shap_image = -v_negpass_filter(shap_values_image)

    # Use the global absolute max as a shared color scale so both panels are comparable.
    shap_vmax = np.max([-shap_values_image.min(), shap_values_image.max()])

    fig, ax = plt.subplots(ncols=2, figsize=(50,50), layout="compressed")

    # Left panel: pixels supporting GOOD classification (class 0 positive SHAP), shown in blue.
    ax[0].imshow(full_image, cmap = 'gray', alpha = 1)
    ax[0].imshow(pos_shap_image[:,:,0], vmin = 0, vmax = shap_vmax, cmap = 'Blues', alpha=0.5)
    ax[0].text(0, 1, "shap pixels supporting GOOD classification", size= size, color = 'mediumblue', weight='bold')
    ax[0].text(0, 3, f"predicted probabilty = {y_prob[0]}%", size= size, color = 'mediumblue', weight='bold')

    # Right panel: pixels supporting DEFECT classification (class 0 negative SHAP = class 1 positive),
    # shown in red.
    ax[1].imshow(full_image, cmap = 'gray', alpha = 1)
    ax[1].imshow(neg_shap_image[:,:,0], vmin = 0, vmax = shap_vmax, cmap = 'Reds', alpha=0.5)
    ax[1].text(0, 1, "shap pixels supporting DEFECT classification", size= size, color = 'brown', weight='bold')
    ax[1].text(0, 3, f"predicted probabilty = {y_prob[1]}%", size= size, color = 'brown', weight='bold')

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_4class_shap_figures_2_chan(shap_values, 
                                     features_test_array, 
                                     y_predict, 
                                     image_num, 
                                     label_dict, 
                                     labels_test, size = 30, 
                                     filename = None):
    '''
    Visualizes SHAP pixel-level explanations for a single image from the 4-class defect
    identification model that uses TWO kernel sizes (e.g. 3x3 and 7x7) in parallel branches.
    Because each branch produces its own SHAP values, shap_values is a list of two arrays.

    Produces two plots:
      1. The raw grayscale image with actual and predicted labels.
      2. A 4-row x 4-column SHAP grid:
           col 0: 3x3 kernel branch — pixels AGAINST this class (blue)
           col 1: 3x3 kernel branch — pixels FOR this class (red)
           col 2: 7x7 kernel branch — pixels AGAINST this class (blue)
           col 3: 7x7 kernel branch — pixels FOR this class (red)
         One row per defect class. Comparing columns 0-1 vs 2-3 reveals which spatial
         scale each kernel size responds to.

    Input:
    shap_values (list of 2 numpy arrays): SHAP values for the two kernel-size branches,
                                          each of shape (n_images, height, width, channels, n_classes).
                                          shap_values[0] = smaller kernel (e.g. 3x3),
                                          shap_values[1] = larger kernel (e.g. 7x7).
    features_test_array (numpy array):   Test images, shape (n_images, height, width, channels).
    y_predict (numpy array):             Model output probabilities, shape (n_images, n_classes).
    image_num (int):                     Index of the image to visualize.
    label_dict (dict):                   Maps class integer codes to class name strings.
    labels_test (list):                  Actual integer class labels for the test set.
    size (int):                          Font size for on-image text annotations.
    filename (str or None):              If provided, saves both plots as .png files.
                                         The raw image is saved with a '_raw.png' suffix.
    '''
    print(f'image_num: {image_num}')

    # Convert raw model output probabilities to percentages rounded to 1 decimal place.
    y_prob = y_predict[image_num]
    y_prob = np.round((y_prob*10000).astype(int)/100, 1)
    print(f'Probabilities: {y_prob}')
    print(f'actual: {labels_test[image_num], label_dict[labels_test[image_num]]}, predicted{y_prob.argmax(), label_dict[y_prob.argmax()]}')

    # --- Plot 1: Raw grayscale image with actual and predicted class labels ---
    plt.figure(figsize = (5,5))
    plt.imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
               vmin = features_test_array[image_num,:,:,0].min(), 
               vmax = features_test_array[image_num,:,:,0].max(),
               alpha = 1)
    plt.text(0, 3, 
             f"actual: '{label_dict[labels_test[image_num]]}'", 
             size= int((size)/2.5), color = 'white', weight='bold')
    plt.text(0, 7, 
             f"predicted: '{label_dict[y_prob.argmax()]}'", 
             size= int((size)/2.5), color = 'white', weight='bold')
    plt.xticks([], fontsize = 0)
    plt.yticks([], fontsize = 0)
    if filename is not None:
        rawfilename = filename.replace(".png", "_raw.png")
        plt.savefig(rawfilename, bbox_inches='tight')
    plt.show()

    # --- Plot 2: SHAP overlay grid (4 classes x 4 columns: two branches x FOR/AGAINST) ---

    # Extract SHAP arrays for each kernel-size branch for this image.
    # _0 corresponds to the smaller kernel branch (e.g. 3x3).
    # _1 corresponds to the larger kernel branch (e.g. 7x7).
    # Shape of each: (height, width, n_classes).
    shap_values_image_0 = shap_values[0][image_num,:,:,0,:]
    shap_values_image_1 = shap_values[1][image_num,:,:,0,:]

    # Normalize each branch independently: subtract mean, then divide by range.
    # This centers the values around zero and puts both branches on a comparable scale,
    # making the FOR/AGAINST overlays visually consistent across branches.
    shap_values_image_0 = shap_values_image_0 - shap_values_image_0.mean()
    shap_values_image_0 = shap_values_image_0 / (shap_values_image_0.max() - shap_values_image_0.min())
    shap_values_image_1 = shap_values_image_1 - shap_values_image_1.mean()
    shap_values_image_1 = shap_values_image_1 / (shap_values_image_1.max() - shap_values_image_1.min())

    # Compute per-branch color scale maxima for the overlays.
    vmax_0 = np.max([-shap_values_image_0.min(), shap_values_image_0.max()])
    vmax_1 = np.max([-shap_values_image_1.min(), shap_values_image_1.max()])
    print(f'[-shap_values_image_0.min(),shap_values_image_0.max()] ={[-shap_values_image_0.min(),shap_values_image_0.max()]}')
    print(f'[-shap_values_image_1.min(),shap_values_image_1.max()] ={[-shap_values_image_1.min(),shap_values_image_1.max()]}')
    print(f'vmax_0 = {vmax_0}, vmax_1 = {vmax_1}')

    fig, ax = plt.subplots(nrows = 4, ncols=4, figsize=(50,50), layout="compressed")
    spaces = ' signal              '
    fig.suptitle(f'                 3x3 kernels negative{spaces}3x3 kernels positive{spaces}7x7 kernels negative{spaces}7x7 kernels positive{spaces}', 
                 fontsize=50)

    # Separate positive and negative SHAP values for each branch.
    pos_shap_values_image_0 = v_pospass_filter(shap_values_image_0)
    neg_shap_values_image_0 = v_negpass_filter(shap_values_image_0)
    pos_shap_values_image_1 = v_pospass_filter(shap_values_image_1)
    neg_shap_values_image_1 = v_negpass_filter(shap_values_image_1)

    # One row per defect class; four columns per row.
    for l in range(4):

        textcoordx = 0
        textcoordy = 15

        # --- Column 0: 3x3 branch, pixels AGAINST this class (negative SHAP, blue) ---
        shap_image = -neg_shap_values_image_0[:,:,l]
        ax[l,0].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        # Row label (class name and predicted probability) rendered vertically on the left margin.
        ax[l,0].text(-10, textcoordy+47, f'"{label_dict[l]}" class \n pred prob = {y_prob[l]}%', 
                     size= size+8, color = 'black', weight='bold', rotation = 90)
        ax[l,0].text(textcoordx, textcoordy, f"3x3 channels \nshap pixels \nAGAINST \n'{label_dict[l]}' classification", size= size, color = 'mediumblue', weight='bold')
        ax[l,0].imshow(shap_image, vmin = 0, vmax = vmax_0, cmap = 'Blues', alpha=0.5)
        ax[l,0].set_xticklabels([], fontsize = 0)
        ax[l,0].set_yticklabels([], fontsize = 0)
        ax[l,0].xaxis.set_ticks_position('none')

        # --- Column 1: 3x3 branch, pixels FOR this class (positive SHAP, red) ---
        shap_image = pos_shap_values_image_0[:,:,l]
        ax[l,1].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,1].text(textcoordx, textcoordy, f"3x3 channels \nshap pixels \nFOR \n'{label_dict[l]}' classification", 
                     size= size, color = 'brown', weight='bold')
        ax[l,1].imshow(shap_image, vmin = 0, vmax = vmax_0, cmap = 'Reds', alpha=0.5)
        ax[l,1].set_xticklabels([], fontsize = 0)
        ax[l,1].set_yticklabels([], fontsize = 0)
        ax[l,1].xaxis.set_ticks_position('none')

        # --- Column 2: 7x7 branch, pixels AGAINST this class (negative SHAP, blue) ---
        shap_image = -neg_shap_values_image_1[:,:,l]
        ax[l,2].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,2].text(textcoordx, textcoordy, f"7x7 channels \nshap pixels \nAGAINST \n'{label_dict[l]}' classification", 
                     size= size, color = 'mediumblue', weight='bold')
        ax[l,2].imshow(shap_image, vmin = 0, vmax = vmax_1, cmap = 'Blues', alpha=0.5)
        ax[l,2].set_xticklabels([], fontsize = 0)
        ax[l,2].set_yticklabels([], fontsize = 0)
        ax[l,2].xaxis.set_ticks_position('none')

        # --- Column 3: 7x7 branch, pixels FOR this class (positive SHAP, red) ---
        shap_image = pos_shap_values_image_1[:,:,l]
        ax[l,3].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,3].text(textcoordx, textcoordy, f"7x7 channels \nshap pixels \nFOR \n'{label_dict[l]}' classification", size= size, color = 'brown', weight='bold')
        ax[l,3].imshow(shap_image, vmin = 0, vmax = vmax_1, cmap = 'Reds', alpha=0.5)
        ax[l,3].set_xticklabels([], fontsize = 0)
        ax[l,3].set_yticklabels([], fontsize = 0)
        ax[l,3].xaxis.set_ticks_position('none')

    if filename is not None:
        print(f'filename = {filename}')
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
