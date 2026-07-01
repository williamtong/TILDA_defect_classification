import random
import os
import numpy as np 
import pandas as pd
import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, BatchNormalization, concatenate, Conv2D, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras 
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
import matplotlib.pyplot as plt

# Note: this import uses a bare module name because the notebooks that call this file
# execute os.chdir() into the functions/ directory before importing, making this the
# working directory at runtime.
from plotting import plot_ROC, plot_2class_histograms, plot_confusion_matrix

# --- Label dictionaries ---
# These map between integer class codes and human-readable defect class names.
# label_dict is used to decode model predictions; revlabel_dict for encoding labels from file paths.
label_dict = {0: 'objects', 1: 'hole', 2: 'oil_spot', 3: 'thread_error'}
revlabel_dict = {'objects': 0, 'hole': 1, 'oil_spot': 2, 'thread_error': 3}

# Augmented versions of the label dictionaries, used when training with augmented image sets.
aug_revlabel_dict = {'objects_augmented': 0, 'hole_augmented': 1, 'oil_spot_augmented': 2, 'thread_error_augmented': 3}
auglabel_dict = {'objects': 'objects_augmented', 'hole': 'hole_augmented', 'oil_spot': 'oil_spot_augmented', 
                 'thread_error': 'thread_error_augmented', 'good': 'good_augmented'}


def get_class_weight(labels):
    '''
    Calculates inverse-frequency class weights to counteract class imbalance during training.
    A class with fewer samples receives a higher weight so the model pays more attention to it.
    For example, if one class has 9000 samples and another has 1000, the weights will be
    0.000111 and 0.001 respectively — the minority class gets 9x more weight.

    Input:
    labels (list):  List of integer class labels for the training data.

    Output:
    class_weights (dict): Dictionary mapping each class integer code to its weight.
    '''
    class_weights = {}
    for label in set(labels):
        print(1/labels.count(label))
        class_weights[label] = 1/labels.count(label)
    return class_weights


def preprocessor(image, div_by_255=False):
    '''
    Loads a JPEG image from disk, decodes it, and resizes it to 64x64 pixels.
    Optionally normalizes pixel values to the range [0, 1] by dividing by 255.
    Note: normalization was found experimentally to worsen model performance in this
    use case (absolute pixel intensity carries meaningful signal), so div_by_255
    defaults to False.

    Input:
    image (str):         File path to the JPEG image.
    div_by_255 (bool):   If True, normalize pixel values to [0, 1].

    Output:
    image (tf.Tensor):   64x64x3 tensor of pixel values.
    '''
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (64, 64))
    if div_by_255:
        image = image/255  
    return image


def map_path_to_augmented(old_feature_paths):
    '''
    Maps a list of original image file paths to their augmented counterparts.
    Each original image has 5 augmented versions, identified by a numeric suffix
    (_monochrome_1.png through _monochrome_5.png).
    The augmented images live in a sibling directory whose name is the original
    class name with '_augmented' appended (e.g. 'hole' -> 'hole_augmented').

    Input:
    old_feature_paths (list of str):  File paths to the original images.

    Output:
    new_features_paths (list of str): File paths to all augmented images
                                      (5x longer than the input list).
    '''
    new_features_paths = []
    for path in old_feature_paths:
        label = path.split("/")[-2]
        newlabel = auglabel_dict[label]
        for counter in range(1, 6):
            new_feature_path = path.replace(label, newlabel).replace('.png', '_monochrome_'+str(counter)+'.png').replace(" ", "_")
            new_features_paths.append(new_feature_path)
    return new_features_paths


def recreate_labels(features, label_dict):
    '''
    Extracts integer class labels from image file paths.
    Assumes the class name is the second-to-last component of the path
    (i.e. the parent directory name), e.g. '.../hole/image.png' -> 1.

    Input:
    features (list of str):  Image file paths.
    label_dict (dict):       Maps class name strings to integer codes.

    Output:
    labels_numerical (list): Integer class label for each image.
    '''
    labels_text = [path.split("/")[-2] for path in features]
    labels_numerical = [label_dict[label_text] for label_text in labels_text]
    return labels_numerical


def compile_image_model(dropout, kernel_dims=[5, 7, 9], nums_filters=[4, 4, 4],
                        patience=20, lr=0.001, num_classes=4,
                        dot_img_file="../images/network/image_learner_model.png"):
    '''
    Builds and compiles a multi-branch CNN model. Each branch uses a different
    convolutional kernel size, allowing the model to capture features at different
    spatial scales simultaneously. The branches are concatenated and fed into a
    shared fully connected output layer.

    Each branch has the same structure: 3 convolutional blocks, each consisting of:
      Conv2D -> BatchNormalization -> Dropout -> MaxPool2D
    The number of filters doubles with each successive block (e.g. 8 -> 16 -> 32).

    The branches are merged via concatenation, followed by a Dense layer and a
    softmax output (or sigmoid for 2-class problems, handled automatically by
    sparse_categorical_crossentropy when num_classes=2).

    Input:
    kernel_dims (list of int):   Kernel sizes for each branch, e.g. [3, 7] for a
                                 two-branch 3x3 and 7x7 model.
    dropout (float):             Dropout rate applied after each BatchNorm layer.
    nums_filters (int or list):  Number of filters in the first conv block of each
                                 branch. If int, the same value is used for all branches.
    patience (int):              Early stopping patience (passed through for reference).
    lr (float):                  Learning rate for the Adam optimizer.
    num_classes (int):           Number of output classes (2 for detection, 4 for identification).
    dot_img_file (str):          File path to save a diagram of the model architecture.

    Output:
    image_learner:               Compiled Keras Model, ready to call .fit() on.
    '''
    # If a single integer is given for nums_filters, broadcast it to all branches.
    if type(nums_filters) == int:
        nums_filters = [nums_filters]*len(kernel_dims)
    kernel_dims = zip(kernel_dims, nums_filters)

    inputs_list = []
    flattened_list = []

    # Build one convolutional branch per kernel size.
    for dim, num_filters in kernel_dims:
        dim_text = '_' + str(dim)

        # Each branch takes the same 64x64 grayscale image as input.
        inputs = Input(shape=(64, 64, 1), name=f'image_input_for {dim_text}x{dim_text[1:]}_kernel_subnetwork')

        # --- Block 1 ---
        conv_1 = Conv2D(filters=int(num_filters), kernel_size=(dim, dim),
                        padding="same", strides=(1, 1),
                        activation="relu", name='2dfilter1'+dim_text)(inputs)
        batch_norm_1 = BatchNormalization(axis=2, name='batchnormalization1'+dim_text)(conv_1)
        dropout_1 = Dropout(dropout, name='dropout1'+dim_text)(batch_norm_1)
        maxpool_1 = MaxPool2D(name='maxpool1'+dim_text)(dropout_1)

        # --- Block 2: double the number of filters ---
        num_filters *= 2
        conv_2 = Conv2D(filters=int(num_filters), kernel_size=(dim, dim),
                        padding="same", strides=(1, 1),
                        activation="relu", name='2dfilter2'+dim_text)(maxpool_1)
        batch_norm_2 = BatchNormalization(axis=2, name='batchnormalization2'+dim_text)(conv_2)
        dropout_2 = Dropout(dropout, name='dropout2'+dim_text)(batch_norm_2)
        maxpool_2 = MaxPool2D(name='maxpool2'+dim_text)(dropout_2)

        # --- Block 3: double again ---
        num_filters *= 2
        conv_3 = Conv2D(filters=int(num_filters), kernel_size=(dim, dim),
                        padding="same", strides=(1, 1),
                        activation="relu", name='2dfilter3'+dim_text)(maxpool_2)
        batch_norm_3 = BatchNormalization(axis=2, name='batchnormalization3'+dim_text)(conv_3)
        dropout_3 = Dropout(dropout, name='dropout3'+dim_text)(batch_norm_3)
        maxpool_3 = MaxPool2D(name='maxpool3'+dim_text)(dropout_3)

        # Flatten the 3D output of the last conv block into a 1D vector for the Dense layer.
        num_filters /= 4
        flattened = Flatten(name='flattened5'+dim_text)(maxpool_3)

        inputs_list.append(inputs)
        flattened_list.append(flattened)

    # Concatenate the flattened outputs of all branches into a single feature vector.
    merged = concatenate(flattened_list, name='concatenate_all_subnetworks')

    # Shared fully connected output block.
    dense_last = Dense(units=100, activation='relu', name='dense_last')(merged)
    outputs = Dense(units=num_classes, activation="softmax", name='softmax_output')(dense_last)

    image_learner = Model(inputs=inputs_list, outputs=outputs, name='image_learner')
    image_learner.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"],
                          optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    print(image_learner.summary())
    # Save a diagram of the full model architecture to file.
    plot_model(image_learner,
               show_layer_activations=True,
               to_file=dot_img_file,
               show_shapes=True,
               show_dtype=True,
               show_layer_names=True)
    return image_learner


class CNN_model:
    '''
    A wrapper class around the multi-branch CNN built by compile_image_model().
    Provides a scikit-learn-style fit/predict interface and automatically plots
    confusion matrices after fitting and prediction.
    '''

    def __init__(self, kernel_dims=[5, 7, 9], dropout=0.1, nums_filters=4,
                 lr=0.001, patience=5, num_classes=4):
        '''
        Instantiates and compiles the CNN model.

        Input:
        kernel_dims (list of int):  Kernel sizes for each parallel branch.
        dropout (float):            Dropout rate applied after each BatchNorm layer.
        nums_filters (int):         Number of filters in the first conv block of each branch.
        lr (float):                 Learning rate for the Adam optimizer.
        patience (int):             Number of epochs with no val_accuracy improvement
                                    before early stopping is triggered.
        num_classes (int):          Number of output classes (2 for detection, 4 for identification).
        '''
        self.kernel_dims = kernel_dims
        self.dropout = dropout
        self.lr = lr
        # Early stopping monitors validation accuracy and restores the best weights when triggered.
        self.call_back = EarlyStopping(monitor="val_accuracy",
                                       verbose=1,
                                       patience=patience,
                                       restore_best_weights=True,
                                       start_from_epoch=5)
        self.num_classes = num_classes
        print(f'self.kernel_dims = {self.kernel_dims}')
        print(f'self.dropout = {self.dropout}')
        self.image_learner = compile_image_model(kernel_dims=self.kernel_dims,
                                                 dropout=self.dropout,
                                                 nums_filters=nums_filters,
                                                 lr=self.lr,
                                                 num_classes=num_classes)

    def fit(self, features_train, labels_train, features_eval, labels_eval, label_dict,
            batch_size=128, class_weight=None, epochs=100000, verbose=2):
        '''
        Trains the CNN model with early stopping monitored on the eval set.
        The same image array is passed once per branch (since all branches share the
        same input image). Plots confusion matrices on the eval set after training.

        Input:
        features_train (numpy array):  Training images, shape (N, 64, 64, 1).
        labels_train (numpy array):    Integer class labels for training images.
        features_eval (numpy array):   Evaluation images, shape (P, 64, 64, 1).
        labels_eval (numpy array):     Integer class labels for evaluation images.
        label_dict (dict):             Maps integer codes to class name strings.
        batch_size (int):              Mini-batch size for stochastic gradient descent.
        class_weight (dict):           Per-class loss weights (see get_class_weight()).
        epochs (int):                  Maximum number of training epochs.
        verbose (int):                 Keras verbosity: 0 = silent, 1 = progress bar, 2 = one line per epoch.

        Output:
        history:        Keras History object containing accuracy and val_accuracy per epoch.
        image_learner:  The trained Keras model.
        '''
        self.label_dict = label_dict

        # The model has one input per branch; all branches receive the same image,
        # so we replicate the array once per kernel size.
        self.history = self.image_learner.fit(
            [features_train]*len(self.kernel_dims), labels_train,
            validation_data=[[features_eval]*len(self.kernel_dims), labels_eval],
            callbacks=[self.call_back],
            epochs=epochs, batch_size=batch_size,
            class_weight=class_weight,
            verbose=verbose
        )

        print(self.image_learner.summary())
        print("Finished fitting.  Predicting X...")
        y_predict = self.image_learner.predict([features_eval]*len(self.kernel_dims))
        y_predict = [self.label_dict[y.argmax()] for y in y_predict]
        print("Finished predicting X with eval data set.")
        y_actual = [self.label_dict[label] for label in labels_eval]

        # Plot confusion matrices in four normalizations for a complete picture of performance.
        plot_confusion_matrix(y_actual, y_predict, normalize=None)   # raw counts
        plot_confusion_matrix(y_actual, y_predict, normalize='all')  # fraction of all samples
        plot_confusion_matrix(y_actual, y_predict, normalize='pred') # precision
        plot_confusion_matrix(y_actual, y_predict, normalize='true') # recall

        return self.history, self.image_learner

    def predict(self, features_test, labels_test=None):
        '''
        Runs inference on a set of images. If actual labels are supplied, plots
        confusion matrices. For 2-class models, also plots the ROC curve and
        predicted probability distributions.

        Input:
        features_test (numpy array):  Test images, shape (Q, 64, 64, 1).
        labels_test (numpy array):    Integer class labels for test images (optional).

        Output:
        y_predict (list):       Predicted class name string for each image.
        y_predict_prob (array): Raw softmax probabilities, shape (Q, num_classes).
        '''
        # Replicate the test array once per branch, matching the multi-input model signature.
        y_predict_prob = self.image_learner.predict([features_test]*len(self.kernel_dims))
        y_predict = [self.label_dict[y.argmax()] for y in y_predict_prob]

        print("Finished predicting X.")

        if labels_test is not None:
            y_actual = [self.label_dict[label] for label in labels_test]

            # Plot confusion matrices in four normalizations.
            plot_confusion_matrix(y_actual, y_predict, normalize=None)   # raw counts
            plot_confusion_matrix(y_actual, y_predict, normalize='all')  # fraction of all samples
            plot_confusion_matrix(y_actual, y_predict, normalize='pred') # precision
            plot_confusion_matrix(y_actual, y_predict, normalize='true') # recall

            # For the 2-class model, also plot ROC curve and probability histograms.
            if self.num_classes == 2:
                plot_ROC(labels_test, y_predict_prob[:,1])
                plot_2class_histograms(labels_test, y_predict_prob, self.label_dict,
                                       stepsize=0.001, density=True, semilog=False, holdout=True)

        return y_predict, y_predict_prob