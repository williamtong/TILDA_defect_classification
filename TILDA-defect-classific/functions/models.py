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

from plotting import plot_ROC, plot_2class_histograms, plot_confusion_matrix

label_dict = {0: 'objects', 1: 'hole', 2: 'oil_spot', 3: 'thread_error'}
revlabel_dict = {'objects': 0, 'hole': 1, 'oil_spot': 2, 'thread_error': 3}

aug_revlabel_dict = {'objects_augmented': 0, 'hole_augmented': 1, 'oil_spot_augmented': 2, 'thread_error_augmented': 3}

auglabel_dict = {'objects': 'objects_augmented', 'hole': 'hole_augmented', 'oil_spot': 'oil_spot_augmented', 
                 'thread_error': 'thread_error_augmented', 'good': 'good_augmented'}

def get_class_weight(labels):
    '''
    This functions calculates the weighting factor for each class to be input into the model. 
    For example if one class has a population of 9000 and the other class has a population of 1000, 
    the weighting factor would be 0.1 for the first class and 0.9 for the second class.

    input:
    labals (list):  A list of the target variable for the data.

    output:
    class weight (dict): dictionary of class weights
    '''
    class_weights = {}
    for label in set(labels):
        print(1/labels.count(label))
        class_weights[label] = 1/labels.count(label)
    return class_weights
    

def preprocessor(image, div_by_255 = False):
    '''
    Function to format image to right size.  
    Optional 
    input:
    image:  any size image

    output:
    image:  64x64 image
    '''
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, (64, 64))
    if div_by_255:
        image = image/255  
    return image


def map_path_to_augmented(old_feature_paths):
    new_features_paths = []
    for path in old_feature_paths:
        label = path.split("/")[-2]
        newlabel = auglabel_dict[label]
        for counter in range(1,6):
            # print(path, newlabel)
            new_feature_path = path.replace(label, newlabel).replace('.png', '_monochrome_'+str(counter)+'.png').replace(" ","_")
            new_features_paths.append(new_feature_path)
    return new_features_paths

    
def recreate_labels(features, label_dict):
    '''This function extracts the labels from the data paths, given the images' individual data paths. '''
    labels_text = [path.split("/")[-2] for path in features]
    labels_numerical = [label_dict[label_text] for label_text in labels_text]
    return labels_numerical



def compile_image_model(dropout, kernel_dims = [5, 7, 9], nums_filters = [4,4,4], 
                        patience = 20, lr =0.001, num_classes =4, 
                        dot_img_file = "../images/network/image_learner_model.png"):
    '''This function instantiation a CNN model with the follow specs
    The CNN model will have three blocks.  For each subsequent block the number of filters doubles.
    The last block is fully connected layer, followed by the output layer which is a softmax function.
    The loss function is "sparse_categorical_crossentropy".
    The optimizer is Adam.

    input:
    kernel_dims (list of integers):   List of the dimensions of the square convolutional kernels (filters).
    dropout (0 ≤ float < 1):          Fraction connections that are dropped out for each block
    nums_filters (list of integers):  List of the numbers of kernels (filters) for the starting block.  
    patience (interger):              Number of epochs with no improvement after which training will be stopped. 
    num_classes (integer):            Number of classes of the target variable

    output:
    image_learner:             Instantiated CNN model
    '''

    if type(nums_filters) == int:
        nums_filters = [nums_filters]*len(kernel_dims)
    kernel_dims = zip(kernel_dims, nums_filters)

    inputs_list = []
    flattened_list = []
    for dim, num_filters in kernel_dims:
        dim_text = '_' +str(dim)
        inputs = Input(shape = (64, 64, 1), name = f'image_input_for {dim_text}x{dim_text[1:]}_kernel_subnetwork')
        conv_1 = Conv2D(filters = int(num_filters), 
                        kernel_size = (dim, dim), 
                        padding="same", strides = (1, 1), 
                        activation = "relu", name = '2dfilter1'+dim_text)(inputs)
        batch_norm_1 = BatchNormalization(axis = 2, name = 'batchnormalization1'+dim_text)(conv_1)
        dropout_1 = Dropout(dropout, name = 'dropout1'+dim_text)(batch_norm_1)
        maxpool_1 = MaxPool2D(name = 'maxpool1'+dim_text)(dropout_1)
        
        num_filters *= 2 #double the number of kernels for each successive block)
        conv_2 = Conv2D(filters = int(num_filters), 
                        kernel_size = (dim, dim), 
                        padding="same",strides = (1, 1), 
                        activation = "relu", name = '2dfilter2'+dim_text)(maxpool_1)
        batch_norm_2 = BatchNormalization(axis = 2, name = 'batchnormalization2'+dim_text)(conv_2)
        dropout_2 = Dropout(dropout, name = 'dropout2'+dim_text)(batch_norm_2)
        maxpool_2 = MaxPool2D(name = 'maxpool2'+dim_text)(dropout_2)
        
        num_filters *= 2
        conv_3 = Conv2D(filters = int(num_filters), 
                        kernel_size = (dim, dim), 
                        padding="same",strides = (1, 1), 
                        activation = "relu", name = '2dfilter3'+dim_text)(maxpool_2)
        batch_norm_3 = BatchNormalization(axis = 2, name = 'batchnormalization3'+dim_text)(conv_3)
        dropout_3 = Dropout(dropout, name = 'dropout3'+dim_text)(batch_norm_3)
        maxpool_3 = MaxPool2D(name = 'maxpool3'+dim_text)(dropout_3)

        # Flattened layer
        num_filters /= 4
        flattened = Flatten(name = 'flattened5'+dim_text)(maxpool_3)

        # Append inputs and outputs to list
        inputs_list.append(inputs)
        flattened_list.append(flattened)

    merged = concatenate(flattened_list, name = 'concatenate_all_subnetworks')

    dense_last = Dense(units = 100, activation = 'relu', name = 'dense_last')(merged)
    outputs = Dense(units = num_classes, activation = "softmax", name = 'softmax_output')(dense_last)
    
    image_learner = Model(inputs = inputs_list, outputs = outputs, name = 'image_learner')
    image_learner.compile(loss = "sparse_categorical_crossentropy", metrics = ["accuracy"], 
                          optimizer = tf.keras.optimizers.Adam(learning_rate= lr))
    dot_img_file = dot_img_file
    print(image_learner.summary())
    plot_model(image_learner,
               show_layer_activations=True,
               to_file=dot_img_file,
               show_shapes=True,
               show_dtype=True,
               show_layer_names=True)
    return image_learner


class CNN_model: 
    ''' This function instantiate an RFM (sklearn) or gbm (xgboost) model and paraments,
    trains a model, and outputs the training scores.
    '''

    def __init__(self, kernel_dims = [5,7,9], dropout = 0.1, nums_filters = 4, 
                 lr = 0.001, patience = 5, num_classes = 4):
        '''
        input:
        filter_dim (integer):      The dimension of a square convolutional filter.
        dropout (0 ≤ float < 1):   Fraction connections that are dropped out for each block
        num_filters (integer):     Number of filter for the starting block.  
        patience (interger):       Number of epochs to elapse  Number of epochs with no improvement after which training will be stopped. 
        num_classes (integer):     Number of classes of the target variable
        '''
        # self.filter_dim = filter_dim,
        self.kernel_dims = kernel_dims
        self.dropout = dropout
        self.lr = lr
        self.call_back = EarlyStopping(monitor="val_accuracy", 
                              verbose=1, 
                              patience = patience, 
                              restore_best_weights = True,
                              start_from_epoch= 5)
        self.num_classes = num_classes
        print(f'self.kernel_dims = {self.kernel_dims}')
        print(f'self.dropout = {self.dropout}')
        self.image_learner = compile_image_model(kernel_dims = self.kernel_dims, 
                                                          dropout = self.dropout, 
                                                          nums_filters = nums_filters, 
                                                          lr = self.lr, 
                                                          num_classes = num_classes)

    def fit(self, features_train, labels_train, features_eval, labels_eval, label_dict, batch_size=128, class_weight=None, epochs = 100000, verbose=2):
        '''
        This uses the image_learner model to fit the data set. It uses the the eval data set to montior overfitting,
        and stops the fitting after "patience" number of epochs with no improvement. 
        Then it uses the eval data set to plot the confusion matrices.
        
        input:
        features_train (numpy array):  An M x M x N array of training set images, N x N are the images' dimension.  There are M images.
        labels_train (numpy array):    N x 1 array of target variable for the training set of the images.    
        features_eval(numpy array):    An M x M x P array of evalution set images, N x N are the images' dimension.  There are P images.
        labels_eval (numpy array):     P x 1 array of target variables for the evalution set of the images. 
        label_dict (dict):             label dictionary to tell the labels are.  For example, {0: "good", 1: "defect"}
        batch_size (integer):          minibatch size for stochastic gradient descent
        class_weight (dict):          See explanation for get_class_weight function.
        epochs (integer):              Max number of epoch if not early stopped.
        verbose (0, 1, 2):             Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action.

        outputs:
        history:        accuracy and validation accuracy
        image_learner:  trained model
        '''
        self.label_dict = label_dict
        self.history = self.image_learner.fit([features_train]*len(self.kernel_dims), labels_train,
                                              validation_data = [[features_eval]*len(self.kernel_dims), labels_eval], 
                                              callbacks = [self.call_back], 
                                              epochs = epochs,  batch_size = batch_size,
                                              class_weight=class_weight,
                                              verbose=verbose
                                             )
        
        print(self.image_learner.summary())   
        print("Finished fitting.  Predicting X...")
        y_predict = self.image_learner.predict([features_eval]*len(self.kernel_dims))
        y_predict = [self.label_dict[y.argmax()] for y in y_predict]
        print("Finished predicting X with eval data set.")
        y_actual = [self.label_dict[label] for label in labels_eval]

        #Plot training results
        plot_confusion_matrix(y_actual, y_predict, normalize = None)
        plot_confusion_matrix(y_actual, y_predict, normalize = 'all')
        plot_confusion_matrix(y_actual, y_predict, normalize = 'pred')
        plot_confusion_matrix(y_actual, y_predict, normalize = 'true')

        return self.history, self.image_learner
        
    def predict(self, features_test, labels_test = None):

        '''
        The uses the image_learner model to predict the images data set.
        If the actual labels are supplied, it will plot the confusion matrices.
        If the num_class == 2, it will plot the AUC-ROC curve and the 2 class probability distribution histograms.

        
        input:
        features_test (numpy array):  An M x M x Q array of training set images, N x N are the images' dimension.  There are Q images.
        labels_test (numpy array):    Q x 1 array of target variable for the training set of the images.    
     
        output:
        y_predict:  The class with the max probability
        y_predict_prob:  The predicted probabilities.
        '''

        y_predict_prob = self.image_learner.predict([features_test]*len(self.kernel_dims))
        y_predict = [self.label_dict[y.argmax()] for y in y_predict_prob]
        
        print("Finished predicting X.")
                                                    
        # If y_actual is not None, then calculate the holdout errors and plot results
        if labels_test is not None:
            y_actual = [self.label_dict[label] for label in labels_test]
            #Ploting holdout results
            plot_confusion_matrix(y_actual, y_predict, normalize = None)
            plot_confusion_matrix(y_actual, y_predict, normalize = 'all')
            plot_confusion_matrix(y_actual, y_predict, normalize = 'pred')
            plot_confusion_matrix(y_actual, y_predict, normalize = 'true')
            if self.num_classes == 2:
                plot_ROC(labels_test, y_predict_prob[:,1])
                plot_2class_histograms(labels_test, y_predict_prob, self.label_dict, stepsize = 0.001, density = True, semilog = False, holdout = True)   
        return y_predict, y_predict_prob