import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


def convert_arr_to_monochrome(features_array):
    '''In our data set, the images arrays are have three colors RGB but they are exactly the sample,
    so two of them are redundant.
    This function takes in an array of images and just retains one of the three "colors".
    
    Input: 
    features_array (numpy array):  A p x m x n x 3 x 1 array in which m and n are the dimensions of the images and p is the number of images (RGB).

    Output:
    output_features_array (numpy array):  A p x m x n x 1 array in which m and n are the dimensions of the images and p is the number of images (now monochrome)

    '''
    num_images = features_array.shape[0]
    output_features_array = np.array([feature_array[:,:,0,:] for feature_array in features_array[range(num_images),:,:,:,:]])
    return output_features_array


def upsample(filepaths, revlabel_dict, label_dict, target_samples_per_class = None, random_seed = 98):
    '''
    This function takes in a set of data (actually their paths) and upsamples to the target number, within rounding approximation.

    Input:
    filepaths (list):                   List of image file paths to be upsampled.
    revlabel_dict:                      Dictionary for reverse class look up. For exmaple, 
                                        {'objects': 0, 'hole': 1, 'oil spot': 2, 'thread error': 3}.
    label_dict:                         Dictionary for class code look up.  For example, 
                                        {0: 'objects', 1: 'hole', 2: 'oil spot', 3: 'thread error'}.
    target_sample_per_class (integer):  The target number the class is to be upsampled.  
                                        (If None, then it is the population of the largest class)
    random_seed (integer):              For seeding random functions if reproducible results are desired.


    Outputs:
    upsampled_filepaths (list):    List of image file paths that are upsampled in randomized ordered.
    upsampled_labels (list):       List of labels for the corresponding image files.
    label_upsample_factors (list): List of upsample factors for each of the classes.
    '''

    labels = [label for label in map(lambda filepath: filepath.split("/")[-2], filepaths)]
    unique_labels = set(labels)
    print(f'unique_labels {unique_labels}')
        
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
    random.Random(random_seed).shuffle(upsampled_filepaths)
    upsampled_labels = [revlabel_dict[label] for label in map(lambda filepath: filepath.split("/")[-2], 
                                                           upsampled_filepaths)]
    return upsampled_filepaths, upsampled_labels, label_upsample_factors 


def create_data_set(all_files, all_labels, label_dict, revlabel_dict, 
                    target_samples_per_class_train = 5120, target_samples_per_class_eval = 512,
                    label_code = 0, random_state = 761):
    '''This function creates a training, eval, and test data set given a set of images.
    It properly stratifies the data based on the label.
    It upsamples each class within each set to a desired value.
    It controls the random set so the data sets are reproducible.

    input:
    all files (list of paths):      List of filepaths of the images.
    all_lables (list of labels):    List of labels for each image.
    label_dict (dict):              Dictionary for the label keys. 
    revlabel_dict (dict):           Reverse lookup dictionary of the label keys.
    target_samples_per_class_train: How many samples to upsample to for each class in the training set.
    target_samples_per_class_eval:  How many samples to upsample to for each class in the eval set.
    label_code (int):               If label_code == 0, it means all labels are in good class for 2-class model, so
                                    all labels should = 0.  Else leave label as is.
    
    '''
    
    features_train, features_eval, labels_train, labels_eval = \
                            train_test_split(all_files, all_labels,
                                             stratify=all_labels, test_size = 0.2, 
                                             random_state = random_state)
    features_test, features_eval, labels_test, labels_eval = \
                                train_test_split(features_eval, labels_eval, 
                                                 stratify=labels_eval, test_size = 0.5, 
                                                 random_state = random_state + 92) #a 
    
    print(len(features_train), len(features_eval), len(features_test))

    if target_samples_per_class_train is not None:
        print(len(features_train), len(features_eval), len(features_test))
        features_train, labels_train, label_upsample_factors =\
            upsample(features_train, revlabel_dict, label_dict, target_samples_per_class = target_samples_per_class_train)
        features_eval, labels_eval, label_upsample_factors =\
            upsample(features_eval, revlabel_dict, label_dict, target_samples_per_class = target_samples_per_class_eval)
        print(len(features_train), len(features_eval), len(features_test))
    
    features_train_array = []
    for feature_path_train in features_train:
        image = plt.imread(feature_path_train)
        # image = image/255
        features_train_array.append(image)
    
    print(f'len(features_train_array): {len(features_train_array)}')
    features_eval_array = []
    for feature_path_eval in features_eval:
        image = plt.imread(feature_path_eval)
        # image = image/255
        features_eval_array.append(image)
        
    features_test_array = []
    for feature_path_test in features_test:
        image = plt.imread(feature_path_test)
        # image = image/255
        features_test_array.append(image)
    
    features_train_array = np.array(features_train_array)
    features_eval_array = np.array(features_eval_array)
    features_test_array = np.array(features_test_array)
    print(f'features_train_array.shape: {features_train_array.shape}')
    print(f'features_test_array.shape: {features_test_array.shape}')
    
    features_train_array = np.expand_dims(features_train_array, -1)
    features_eval_array = np.expand_dims(features_eval_array, -1)
    features_test_array = np.expand_dims(features_test_array, -1)
    
    features_train_array = convert_arr_to_monochrome(features_train_array)
    features_eval_array = convert_arr_to_monochrome(features_eval_array)
    features_test_array = convert_arr_to_monochrome(features_test_array)
    print(f'features_train_array.shape: {features_train_array.shape}')
    print(f'features_test_array.shape: {features_test_array.shape}')

    # label_code == 0 means are labels are in the good class for the 2-class model, 
    # so all labels should be set to 0.
    if label_code == 0:
        labels_train = [label_code]*len(features_train)
        labels_eval = [label_code]*len(features_eval)
        labels_test = [label_code]*len(features_test)
    
    return features_train_array, features_eval_array, features_test_array, labels_train, labels_eval, labels_test


def randomize(features, labels, random_seed = 98):
    '''This function randomizes the order of the features and labels together as pairs.
    Inputs:
    features (list):   List of features.
    '''
    print(len(features), len(labels))
    zipped_list = [x for x in zip(features, labels)]

    random.Random(random_seed).shuffle(zipped_list)
    features = [element[0] for element in zipped_list]
    labels = [element[1] for element in zipped_list]
    return features, labels