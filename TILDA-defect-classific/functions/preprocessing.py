import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


def convert_arr_to_monochrome(features_array):
    '''
    Converts a batch of RGB images to monochrome (single channel) by discarding
    two of the three color channels. In this dataset the images are grayscale stored
    as RGB, meaning all three channels are identical — so two are simply redundant.

    A standard library call like tf.image.rgb_to_grayscale was not used here because
    some libraries automatically rescale pixel values during color conversion, which
    would destroy the absolute intensity information that the model relies on.
    This function avoids that by directly slicing one channel.

    Input:
    features_array (numpy array):  Shape (p, m, n, 3, 1) where p = number of images,
                                   m x n = image dimensions, 3 = RGB channels, 1 = trailing dim
                                   added by np.expand_dims before this call.

    Output:
    output_features_array (numpy array):  Shape (p, m, n, 1) — same images, now single-channel.
    '''
    num_images = features_array.shape[0]
    # Slice channel index 0 (R), discarding the identical G and B channels.
    # The trailing dimension (:) preserves the channel axis so the output is (p, m, n, 1).
    output_features_array = np.array([feature_array[:,:,0,:] for feature_array in features_array[range(num_images),:,:,:,:]])
    return output_features_array


def upsample(filepaths, revlabel_dict, label_dict, target_samples_per_class=None, random_seed=98):
    '''
    Upsamples minority classes by repeating their file paths until each class
    reaches the target count. This is done at the path level (before images are
    loaded into memory) to keep memory usage low.

    Upsampling is preferred over class weighting alone because with very small
    minority classes (e.g. only 337 hole samples), each mini-batch during SGD
    would otherwise contain so few minority samples that gradient updates become
    highly noisy. Upsampling ensures a consistent class distribution per batch.

    Input:
    filepaths (list):                  File paths of the images to upsample.
                                       The class label is inferred from the parent
                                       directory name (second-to-last path component).
    revlabel_dict (dict):              Maps class name strings to integer codes,
                                       e.g. {'objects': 0, 'hole': 1, ...}.
    label_dict (dict):                 Maps integer codes to class name strings,
                                       e.g. {0: 'objects', 1: 'hole', ...}.
    target_samples_per_class (int):    Target number of samples per class after upsampling.
                                       If None, uses the size of the largest class (no downsampling).
    random_seed (int):                 Random seed for reproducible shuffling.

    Output:
    upsampled_filepaths (list):        Upsampled and shuffled file paths.
    upsampled_labels (list):           Integer class labels corresponding to each path.
    label_upsample_factors (dict):     The multiplication factor applied to each class,
                                       keyed by integer class code.
    '''
    # Extract class labels from the parent directory name of each file path.
    labels = [label for label in map(lambda filepath: filepath.split("/")[-2], filepaths)]
    unique_labels = set(labels)
    print(f'unique_labels {unique_labels}')

    # Count how many samples exist per class before upsampling.
    label_counts = {}
    for label in unique_labels:
        label_counts[label] = labels.count(label)

    # Default target: match the size of the largest class.
    if target_samples_per_class is None:
        target_samples_per_class = np.max(list(label_counts.values()))

    # Compute the integer repetition factor for each class.
    label_upsample_factors = {}
    for label in unique_labels:
        label_upsample_factors[revlabel_dict[label]] = target_samples_per_class/label_counts[label]

    # Repeat each class's file paths by its upsample factor, then combine.
    upsampled_filepaths = []
    for label in label_upsample_factors.keys():
        label_filefaths = [label_filepath for label_filepath in filepaths if label_dict[label] in label_filepath]
        # Repeat the list of paths (integer repetition of a Python list).
        label_filefaths = int(np.round(label_upsample_factors[label]))*label_filefaths
        upsampled_filepaths.extend(label_filefaths)

    # Shuffle so classes are interleaved rather than grouped, giving SGD a balanced view.
    random.Random(random_seed).shuffle(upsampled_filepaths)

    # Re-derive labels from the shuffled paths to keep labels and paths in sync.
    upsampled_labels = [revlabel_dict[label] for label in map(lambda filepath: filepath.split("/")[-2],
                                                              upsampled_filepaths)]
    return upsampled_filepaths, upsampled_labels, label_upsample_factors


def create_data_set(all_files, all_labels, label_dict, revlabel_dict,
                    target_samples_per_class_train=5120, target_samples_per_class_eval=512,
                    label_code=0, random_state=761):
    '''
    Builds train, eval, and test sets from a flat list of image file paths and labels.
    The pipeline is:
      1. Stratified split into train (80%) and a temporary holdout (20%).
      2. The holdout is split 50/50 into eval and test sets.
      3. Train and eval sets are upsampled to balance classes for SGD stability.
         The test set is left untouched to give an unbiased holdout evaluation.
      4. Images are loaded from disk into numpy arrays.
      5. A trailing channel dimension is added, then the RGB channels are collapsed
         to monochrome (since all three channels are identical in this dataset).

    The eval set is used during training for early stopping only — it is not the
    final holdout. The test set is used once after training for final evaluation.

    Input:
    all_files (list):                   File paths of all images.
    all_labels (list):                  Integer class labels for each image.
    label_dict (dict):                  Maps integer codes to class name strings.
    revlabel_dict (dict):               Maps class name strings to integer codes.
    target_samples_per_class_train (int): Target upsampled size per class in the train set.
    target_samples_per_class_eval (int):  Target upsampled size per class in the eval set.
    label_code (int):                   For the 2-class detection model, all "good" images
                                        share label 0. Set label_code=0 to override all
                                        labels to 0 (used when loading the "good" class only).
                                        Set to any non-zero value to keep original labels.

    Output:
    features_train_array (numpy array): Training images, shape (N, m, n, 1).
    features_eval_array (numpy array):  Eval images, shape (P, m, n, 1).
    features_test_array (numpy array):  Test images, shape (Q, m, n, 1).
    labels_train (list):                Integer labels for training images.
    labels_eval (list):                 Integer labels for eval images.
    labels_test (list):                 Integer labels for test images.
    '''
    # --- Step 1 & 2: Stratified train / eval / test split ---
    # First split: 80% train, 20% temporary holdout. Stratified to preserve class ratios.
    features_train, features_eval, labels_train, labels_eval = \
        train_test_split(all_files, all_labels,
                         stratify=all_labels, test_size=0.2,
                         random_state=random_state)

    # Second split: divide the 20% holdout evenly into eval (10%) and test (10%).
    features_test, features_eval, labels_test, labels_eval = \
        train_test_split(features_eval, labels_eval,
                         stratify=labels_eval, test_size=0.5,
                         random_state=random_state + 92)

    print(len(features_train), len(features_eval), len(features_test))

    # --- Step 3: Upsample train and eval to balance classes ---
    # The test set is deliberately NOT upsampled — it must reflect the true class distribution.
    if target_samples_per_class_train is not None:
        print(len(features_train), len(features_eval), len(features_test))
        features_train, labels_train, label_upsample_factors = \
            upsample(features_train, revlabel_dict, label_dict,
                     target_samples_per_class=target_samples_per_class_train)
        features_eval, labels_eval, label_upsample_factors = \
            upsample(features_eval, revlabel_dict, label_dict,
                     target_samples_per_class=target_samples_per_class_eval)
        print(len(features_train), len(features_eval), len(features_test))

    # --- Step 4: Load images from disk into numpy arrays ---
    # Note: pixel values are NOT divided by 255 here. Experiments showed that
    # normalizing pixel intensity to [0,1] worsened performance, likely because
    # absolute intensity carries meaningful signal about defect depth/height.
    features_train_array = []
    for feature_path_train in features_train:
        image = plt.imread(feature_path_train)
        features_train_array.append(image)

    print(f'len(features_train_array): {len(features_train_array)}')

    features_eval_array = []
    for feature_path_eval in features_eval:
        image = plt.imread(feature_path_eval)
        features_eval_array.append(image)

    features_test_array = []
    for feature_path_test in features_test:
        image = plt.imread(feature_path_test)
        features_test_array.append(image)

    features_train_array = np.array(features_train_array)
    features_eval_array = np.array(features_eval_array)
    features_test_array = np.array(features_test_array)
    print(f'features_train_array.shape: {features_train_array.shape}')
    print(f'features_test_array.shape: {features_test_array.shape}')

    # --- Step 5: Add trailing channel dim, then collapse RGB to monochrome ---
    # np.expand_dims adds a trailing dimension: (p, m, n) -> (p, m, n, 1).
    # convert_arr_to_monochrome then slices one RGB channel: (p, m, n, 3, 1) -> (p, m, n, 1).
    # The two-step approach avoids library rescaling that would corrupt absolute pixel values.
    features_train_array = np.expand_dims(features_train_array, -1)
    features_eval_array = np.expand_dims(features_eval_array, -1)
    features_test_array = np.expand_dims(features_test_array, -1)

    features_train_array = convert_arr_to_monochrome(features_train_array)
    features_eval_array = convert_arr_to_monochrome(features_eval_array)
    features_test_array = convert_arr_to_monochrome(features_test_array)
    print(f'features_train_array.shape: {features_train_array.shape}')
    print(f'features_test_array.shape: {features_test_array.shape}')

    # For the 2-class detection model, when loading only the "good" class,
    # all labels must be set to 0 regardless of the original label values.
    if label_code == 0:
        labels_train = [label_code]*len(features_train)
        labels_eval = [label_code]*len(features_eval)
        labels_test = [label_code]*len(features_test)

    return features_train_array, features_eval_array, features_test_array, labels_train, labels_eval, labels_test


def randomize(features, labels, random_seed=98):
    '''
    Shuffles features and labels together as matched pairs, preserving the
    correspondence between each image and its label.

    Input:
    features (list):    List of image file paths or arrays.
    labels (list):      List of integer class labels, one per feature.
    random_seed (int):  Random seed for reproducibility.

    Output:
    features (list):    Shuffled features.
    labels (list):      Labels shuffled in the same order as features.
    '''
    print(len(features), len(labels))
    # Zip features and labels together so they are shuffled as pairs.
    zipped_list = [x for x in zip(features, labels)]
    random.Random(random_seed).shuffle(zipped_list)
    features = [element[0] for element in zipped_list]
    labels = [element[1] for element in zipped_list]
    return features, labels