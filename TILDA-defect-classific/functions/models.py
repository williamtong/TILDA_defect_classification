import random
import os
# import glob
import numpy as np 
import pandas as pd
# import seaborn as sns
import tensorflow as tf
# from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras 
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
import matplotlib.pyplot as plt

label_dict = {0: 'objects', 1: 'hole', 2: 'oil spot', 3: 'thread error'}
revlabel_dict = {'objects': 0, 'hole': 1, 'oil spot': 2, 'thread error': 3}

def get_class_weight(labels):
    class_weights = {}
    for label in set(labels):
        print(1/labels.count(label))
        class_weights[label] = 1/labels.count(label)
    return class_weights


def upsample(filepaths, revlabel_dict, label_dict, target_samples_per_class = None):

    labels = [label for label in map(lambda filepath: filepath.split("/")[-2], filepaths)]
    unique_labels = set(labels)
        
    label_counts = {}
    for label in unique_labels:
        label_counts[label] = labels.count(label)
        
    if target_samples_per_class is None:
        target_samples_per_class = np.max(list(label_counts.values()))
        
    label_upsample_factors = {}
    for label in unique_labels:
        label_upsample_factors[revlabel_dict[label]] = target_samples_per_class/label_counts[label]
        
    upsampled_filepaths = []
    for label in label_upsample_factors.keys():
        label_filefaths = [label_filepath for label_filepath in filepaths if label_dict[label] in label_filepath]
        label_filefaths = int(np.round(label_upsample_factors[label]))*label_filefaths
        upsampled_filepaths.extend(label_filefaths)
        # print(f'label = {label}, len(upsampled_filepaths) =  {len(upsampled_filepaths)}')
    random.shuffle(upsampled_filepaths)
    upsampled_labels = [revlabel_dict[label] for label in map(lambda filepath: filepath.split("/")[-2], 
                                                           upsampled_filepaths)]
    return upsampled_filepaths, upsampled_labels, label_upsample_factors 


def preprocessor(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, (64, 64))
    # image = image/255
    return image


def plot_confusion_matrix(y_actual, y_pred, normalize = 'pred'):

    # y_actual = map(lambda label: labels_dict[label], y_actual)
    # y_pred = map(lambda label: labels_dict[label], y_pred)
    cm = confusion_matrix(y_true=y_actual, y_pred=y_pred)
    labels = sorted(list(set(y_actual)))
    print(cm)
    mc_df = pd.DataFrame(cm,
                         index=labels, 
                         columns=labels 
                        )
    plt.title("Confusion Matrix (Counts)")
    sns.heatmap(mc_df, annot =True, 
                # fmt="d",
                cmap=plt.get_cmap('Reds'))
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.xlabel('y_pred')
    plt.ylabel('y_actual')
    plt.show()
    
    if normalize == 'true':
        plt.title("Confusion Matrix (Recall)")
    elif normalize == 'pred':
        plt.title("Confusion Matrix (Precision)")
    elif normalize == 'all':
        plt.title("Confusion Matrix (Accuracy)")
    else:
        return
        
    print('')
    cm = confusion_matrix(y_true=y_actual, y_pred=y_pred, normalize=normalize)
    mc_df = pd.DataFrame(cm,
                         index=labels, 
                         columns=labels
                        )
    sns.heatmap(mc_df, annot =True, 
                # fmt="d",
                color = 'r',
                cmap=plt.get_cmap('Blues'))
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.xlabel('y_pred')
    plt.ylabel('y_actual')
    plt.show()
    return

def recreate_labels(features, label_dict):
    labels_text = [path.split("/")[-2] for path in features]
    labels_numerical = [label_dict[label_text] for label_text in labels_text]
    return labels_numerical


def compile_image_model(filter_dim, dropout, conv_factor =1, patience = 20, lr =0.001, num_classes =5):
    call_back = EarlyStopping(patience = patience, restore_best_weights = True)
    image_learner = Sequential()
    
    image_learner.add(Conv2D(filters = 16 * conv_factor, kernel_size = (filter_dim, filter_dim), 
                             padding="same",strides = (1, 1), activation = "relu"))
    image_learner.add(Dropout(dropout))
    image_learner.add(BatchNormalization(axis = 3, name = 'bn0'))
    # image_learner.add(MaxPool2D())
    
    image_learner.add(Conv2D(filters = 32 * conv_factor, kernel_size = (filter_dim, filter_dim), 
                             padding="same",strides = (1, 1), activation = "relu"))
    image_learner.add(Dropout(dropout))
    image_learner.add(MaxPool2D())
    
    image_learner.add(Conv2D(filters = 64 * conv_factor, kernel_size = (filter_dim, filter_dim), 
                             padding="same", strides = (1, 1), activation = "relu"))
    image_learner.add(Dropout(dropout))
    image_learner.add(MaxPool2D())
    
    # image_learner.add(Conv2D(filters = 128 * conv_factor, kernel_size = (filter_dim, filter_dim),
    #                          padding="same", strides = (1, 1), activation = "relu"))
    # image_learner.add(Dropout(dropout))
    # image_learner.add(MaxPool2D())
    
    image_learner.add(Flatten())
    image_learner.add(Dense(units = 8, activation = "relu"))
    image_learner.add(Dense(units = num_classes, activation = "softmax"))
    
    image_learner.compile(loss = "sparse_categorical_crossentropy", metrics = ["accuracy"], 
                          optimizer = tf.keras.optimizers.Adam(learning_rate= lr))
    print(image_learner.summary)
    return image_learner

class CNN_model: 
    ''' This function instantiate an RFM (sklearn) or gbm (xgboost) model and paraments,
    trains a model, and and outputs the training scores.
    '''

    def __init__(self, filter_dim, dropout, patience = 5, conv_factor = 1, lr = 0.001, num_classes = 5):
        '''
        input:
        Model (sklearn.ensemble or xgboost class): a tree tree.
        params (dict):  parameters to be input into the model
        price_max (int or float): max for the output plot (only).  Not used in model and culling data.
        '''
        self.filter_dim = filter_dim,  
        self.dropout = dropout
        self.conv_factor = conv_factor
        self.lr = lr
        self.call_back = EarlyStopping(monitor="val_accuracy", 
                              verbose=1, 
                              patience = patience, 
                              restore_best_weights = True,
                              start_from_epoch= 10)
        self.image_learner = compile_image_model(filter_dim, dropout, conv_factor = 1, lr = self.lr, num_classes =num_classes)
        # return self

    def fit(self, train_set, eval_set, label_dict, class_weights=None, epochs = 100000, verbose='auto'):
        '''
        input:
        X_train (pandas Dataframe): X data for training
        y_train_actual (pandas Dataframe or numpy array (1d): y data for training
        
        output:
        model: trained model
        '''
        self.label_dict = label_dict
        self.history = self.image_learner.fit(train_set, 
                                              validation_data = [eval_set], 
                                              callbacks = [self.call_back], 
                                              epochs = epochs, 
                                              class_weight=class_weights,
                                              verbose=verbose
                                             )
        
        #Calculate Training error.    
        print("Finished fitting.  Predicting X...")
        images_eval = [image for image, label in eval_set.as_numpy_iterator()][0]
        y_predict = self.image_learner.predict(images_eval)
        # print(f'1 y_predict = {y_predict}')
        y_predict = [self.label_dict[y.argmax()] for y in y_predict]
        # print(f'2 y_predict = {y_predict}')
        print("Finished predicting X.")
        y_actual = [label for image, label in eval_set.as_numpy_iterator()][0]
        # print(f'1 y_actual = {y_actual}')
        y_actual = [self.label_dict[label] for label in y_actual]
        # print(f'2 y_actual = {y_actual}')

        #Plot training results
        plot_confusion_matrix(y_actual, y_predict, normalize = 'pred')

        
        return self.history, self.image_learner
        
    def predict(self, test_set):

        '''
        input:
        X (pandas Dataframe):  X data to be predicted.
                               If there's only one row, input in the following format.
                               X.iloc[row_number, row_number + 1, :]
        y (pandas Dataframe or np array:  y data

        output:
        y_predict
        '''
        # If y_actual is not None, then calculate the holdout errors and plot results

        print("Finished fitting.  Predicting X...")
        images_test = [image for image, label in test_set.as_numpy_iterator()][0]
        y_predict_prob = self.image_learner.predict(images_test)
        # print(f'1 y_predict = {y_predict}')
        y_predict = [self.label_dict[y.argmax()] for y in y_predict_prob]
        # print(f'2 y_predict = {y_predict}')
        print("Finished predicting X.")
        y_actual = [label for image, label in test_set.as_numpy_iterator()][0]
        # print(f'1 y_actual = {y_actual}')
        y_actual = [self.label_dict[label] for label in y_actual]
        # print(f'2 y_actual = {y_actual}')
        #Ploting holdout results
        plot_confusion_matrix(y_actual, y_predict, normalize = 'pred')
        return y_predict, y_predict_prob