import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix, roc_curve, auc, RocCurveDisplay
import shap

from models import get_class_weight, preprocessor, plot_confusion_matrix, compile_image_model, \
                    CNN_model,label_dict, revlabel_dict, recreate_labels, map_path_to_augmented

from preprocessing import upsample

def using_where_neg(x):
    return np.where(x < 0, 0, x)
v_pospass_filter = np.vectorize(using_where_neg)

def using_where_pos(x):
    return np.where(x > 0, 0, x)
v_negpass_filter = np.vectorize(using_where_pos)


def plot_4class_shap_figures(shap_values, 
                             features_test_array, 
                             y_predict, 
                             image_num, 
                             label_dict, 
                             labels_test, size = 30, 
                             # power = 1, 
                             filename = None):
    print(f'image_num: {image_num}')
    y_prob = y_predict[image_num]
    y_prob = np.round((y_prob*10000).astype(int)/100,1)
    print(f'Probabilities: {y_prob}')
    print(f'actual: {labels_test[image_num], label_dict[labels_test[image_num]]}, predicted{y_prob.argmax(), label_dict[y_prob.argmax()]}')
    
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
    # print(rawfilename)
    if filename is not None:
        rawfilename = filename.replace(".png", "_raw.png")
        plt.savefig(rawfilename, bbox_inches='tight')
    plt.show()
    
    # Select specific shap image
    shap_values_image = shap_values[image_num,:,:,0,:]
    
    vmax = np.max([-shap_values_image.min(),shap_values_image.max()])
    fig, ax = plt.subplots(nrows = 4, ncols=2, figsize=(50,50), layout="compressed")
    
    # Remove negative shap values, which stand for negative impact.
    pos_shap_values_image = v_pospass_filter(shap_values_image)
    neg_shap_values_image = v_negpass_filter(shap_values_image)
    for l in range(4):
        shap_image = pos_shap_values_image[:,:,l]
        # vmin, vmax = shap_image.min(), shap_image.max()
        ax[l,1].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,1].text(0, 3, f"shap pixels FOR '{label_dict[l]}' classification", size= size, color = 'brown', weight='bold')
        ax[l,1].text(0, 7, f"predicted probabilty = {y_prob[l]}%", size= size, color = 'brown', weight='bold')
        ax[l,1].imshow(shap_image, vmin = 0, vmax = vmax,
                   cmap = 'Reds', alpha=0.5)
        ax[l,1].set_xticklabels([], fontsize = 0)
        ax[l,1].set_yticklabels([], fontsize = 0)
        ax[l,1].xaxis.set_ticks_position('none') 
    
        shap_image = -neg_shap_values_image[:,:,l]
        ax[l,0].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,0].text(0, 3, f"shap pixels AGAINST '{label_dict[l]}' classification", size= size, color = 'mediumblue', weight='bold')
        # ax[l,0].text(0, 7, f"predicted probabilty = {y_prob[l]}%", size= size, color = 'mediumblue', weight='bold')
        ax[l,0].imshow(shap_image, vmin = 0, vmax = vmax,
                   cmap = 'Blues', alpha=0.5)
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
    
    print(f'image_num: {image_num}')
    y_prob = y_predict[image_num]
    y_prob = np.round((y_prob*10000).astype(int)/100,1)
    print(f'Probabilities: {y_prob}')
    print(f'actual: {labels_test[image_num], label_dict[labels_test[image_num]]}, \
    predicted{y_predict[image_num].argmax(), label_dict[y_predict[image_num].argmax()]}')
    shap_values_image = shap_values[image_num,:,:,0,:]
    
    full_image = features_test_array[image_num,:,:,0]
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
    
    pos_shap_image = v_pospass_filter(shap_values_image)
    neg_shap_image = -v_negpass_filter(shap_values_image)
    shap_vmax = np.max([-shap_values_image.min(),shap_values_image.max()])
    fig, ax = plt.subplots(ncols=2, figsize=(50,50), layout="compressed")

    ax[0].imshow(full_image, cmap = 'gray', 
               alpha = 1),  
    # shap_image = pos_shap_image[:,:,0]
    ax[0].imshow(pos_shap_image[:,:,0], vmin = 0, vmax = shap_vmax,
               cmap = 'Blues', alpha=0.5)    
    ax[0].text(0, 1, "shap pixels supporting GOOD classification", size= size, color = 'mediumblue', weight='bold')
    ax[0].text(0, 3, f"predicted probabilty = {y_prob[0]}%", size= size, color = 'mediumblue', weight='bold')
    
    ax[1].imshow(full_image, cmap = 'gray', 
               alpha = 1),
    ax[1].imshow(neg_shap_image[:,:,0], vmin = 0, vmax = shap_vmax,
               cmap = 'Reds', alpha=0.5) 
    ax[1].text(0, 1, "shap pixels supporting DEFECT classification", size= size, color = 'brown', weight='bold')
    ax[1].text(0, 3, f"predicted probabilty = {y_prob[1]}%", size= size, color = 'brown', weight='bold')
    
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def using_where_neg(x):
    return np.where(x < 0, 0, x)
v_pospass_filter = np.vectorize(using_where_neg)


def using_where_pos(x):
    return np.where(x > 0, 0, x)
v_negpass_filter = np.vectorize(using_where_pos)

def plot_4class_shap_figures_2_chan(shap_values, 
                                     features_test_array, 
                                     y_predict, 
                                     image_num, 
                                     label_dict, 
                                     labels_test, size = 30, 
                                     # power = 1, 
                                     filename = None):
    print(f'image_num: {image_num}')
    y_prob = y_predict[image_num]
    y_prob = np.round((y_prob*10000).astype(int)/100,1)
    print(f'Probabilities: {y_prob}')
    print(f'actual: {labels_test[image_num], label_dict[labels_test[image_num]]}, predicted{y_prob.argmax(), label_dict[y_prob.argmax()]}')
    
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
    # print(rawfilename)
    if filename is not None:
        rawfilename = filename.replace(".png", "_raw.png")
        plt.savefig(rawfilename, bbox_inches='tight')
    plt.show()
    
    # Select specific shap image.
    # The image file is a list of two shap image files, representing the two branches of kernels of two sizes.
    shap_values_image_0 = shap_values[0][image_num,:,:,0,:]
    shap_values_image_1 = shap_values[1][image_num,:,:,0,:]

    shap_values_image_0 = shap_values_image_0-shap_values_image_0.mean()
    shap_values_image_0 = shap_values_image_0/(shap_values_image_0.max() - shap_values_image_0.min())
    shap_values_image_1 = shap_values_image_1-shap_values_image_1.mean()
    shap_values_image_1 = shap_values_image_1/(shap_values_image_1.max() - shap_values_image_1.min())
    
    # vmax_0 = np.max([-shap_values_image_0.min(),shap_values_image_0.max()])
    # vmax_1 = np.max([-shap_values_image_1.min(),shap_values_image_1.max()])
    vmax_0 = np.max([-shap_values_image_0.min(),shap_values_image_0.max()])
    vmax_1 = np.max([-shap_values_image_1.min(),shap_values_image_1.max()])
    print(f'[-shap_values_image_0.min(),shap_values_image_0.max()] ={[-shap_values_image_0.min(),shap_values_image_0.max()]}')
    print(f'[-shap_values_image_1.min(),shap_values_image_1.max()] ={[-shap_values_image_1.min(),shap_values_image_1.max()]}')
    print(f'vmax_0 = {vmax_0}, vmax_1 = {vmax_1}')
    fig, ax = plt.subplots(nrows = 4, ncols=4, figsize=(50,50), layout="compressed")
    spaces = ' signal              '
    fig.suptitle(f'                 3x3 kernels negative{spaces}3x3 kernels positive{spaces}7x7 kernels negative{spaces}7x7 kernels positive{spaces}', 
                 fontsize=50)
    
    # Separate Postive and Negative shap values (negative shap values stand for negative impact
    pos_shap_values_image_0 = v_pospass_filter(shap_values_image_0)
    neg_shap_values_image_0 = v_negpass_filter(shap_values_image_0)
    pos_shap_values_image_1 = v_pospass_filter(shap_values_image_1)
    neg_shap_values_image_1 = v_negpass_filter(shap_values_image_1)

    for l in range(4):
        shap_image = -neg_shap_values_image_0[:,:,l]

        textcoordx = 0
        textcoordy = 15
        ax[l,0].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)  
        ax[l,0].text(-10, textcoordy+47, f'"{label_dict[l]}" class \n pred prob = {y_prob[l]}%', 
                     size= size+8, color = 'black', weight='bold', 
                     # va = 'center'
                     rotation = 90, 
                     # rotation_mode = 'anchor'
                    )
        ax[l,0].text(textcoordx, textcoordy, f"3x3 channels \nshap pixels \nAGAINST \n'{label_dict[l]}' classification", size= size, color = 'mediumblue', weight='bold')
        # ax[l,0].text(0, 7, f"predicted probabilty = {y_prob[l]}%", size= size, color = 'mediumblue', weight='bold')
        ax[l,0].imshow(shap_image, vmin = 0, vmax = vmax_0,
                   cmap = 'Blues', alpha=0.5)
        ax[l,0].set_xticklabels([], fontsize = 0)
        ax[l,0].set_yticklabels([], fontsize = 0)
        ax[l,0].xaxis.set_ticks_position('none') 

        shap_image = pos_shap_values_image_0[:,:,l]
        # vmin, vmax = shap_image.min(), shap_image.max()
        ax[l,1].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,1].text(textcoordx, textcoordy, f"3x3 channels \nshap pixels \nFOR \n'{label_dict[l]}' classification", 
                     size= size, color = 'brown', weight='bold')
        # ax[l,1].text(0, 7, f"predicted probabilty = {y_prob[l]}%", size= size, color = 'brown', weight='bold')
        ax[l,1].imshow(shap_image, vmin = 0, vmax = vmax_0,
                   cmap = 'Reds', alpha=0.5)
        ax[l,1].set_xticklabels([], fontsize = 0)
        ax[l,1].set_yticklabels([], fontsize = 0)
        ax[l,1].xaxis.set_ticks_position('none') 

        shap_image = -neg_shap_values_image_0[:,:,l]
        ax[l,2].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,2].text(textcoordx, textcoordy, f"7x7 channels \nshap pixels \nAGAINST \n'{label_dict[l]}' classification", 
                     size= size, color = 'mediumblue', weight='bold')
        # ax[l,0].text(0, 7, f"predicted probabilty = {y_prob[l]}%", size= size, color = 'mediumblue', weight='bold')
        ax[l,2].imshow(shap_image, vmin = 0, vmax = vmax_1,
                   cmap = 'Blues', alpha=0.5)
        ax[l,2].set_xticklabels([], fontsize = 0)
        ax[l,2].set_yticklabels([], fontsize = 0)
        ax[l,2].xaxis.set_ticks_position('none') 
        
        shap_image = pos_shap_values_image_1[:,:,l]
        # vmin, vmax = shap_image.min(), shap_image.max()
        ax[l,3].imshow(features_test_array[image_num,:,:,0], cmap = 'gray', 
                   vmin = features_test_array[image_num,:,:,0].min(), vmax = features_test_array[image_num,:,:,0].max(),
                   alpha = 1)
        ax[l,3].text(textcoordx, textcoordy, f"7x7 channels \nshap pixels \nFOR \n'{label_dict[l]}' classification", size= size, color = 'brown', weight='bold')
        # ax[l,3].text(0, 7, f"predicted probabilty = {y_prob[l]}%", size= size, color = 'brown', weight='bold')
        ax[l,3].imshow(shap_image, vmin = 0, vmax = vmax_1,
                   cmap = 'Reds', alpha=0.5)
        ax[l,3].set_xticklabels([], fontsize = 0)
        ax[l,3].set_yticklabels([], fontsize = 0)
        ax[l,3].xaxis.set_ticks_position('none') 
    
    if filename is not None:
        print(f'filename = {filename}')
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    