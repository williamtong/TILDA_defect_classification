**Defect detection and classification on industrial fabric**

**Top line results:**

1.  I trained two CNN models, one to detect defects, another to identify what class the defect is.
2.  The best 2-class defect detection model (D-1, with 7x7 kernels and 8 x 2<sup>n</sup> kernels per block) has an ROC-AUC of **91.7%**, a Recall for the _good_ class of 96.1%, and a Recall of 61.7% for the _defect_ class at thresholds of 50%.   The model D-4 has a higher ROC-AUC of 93.1% and may potentially have a better combination of the Precision and Recall values _at different threshold values_.   However, that depends on the business case.  For our purpose, we will employ a 50% threshold.
3.  The best 4-class defect identification model (c-2, with 7x7 kernels and 16 x 2<sup>n</sup> kernels per block) has an accuracy of **63.0%**, against a baseline of 25% (4 classes).  For this  4-class defect identification model, we achieve recall of 41.2%, 65.4%, 55.5%, 79.0% for the defect classes of hole, objects, oil spot, thread error, respectively.  Note there are 4 total classes so the baseline for a model that does "random guessing" is _25%_.  Thus the model is able extract signal.   In particular, this model performs very well for objects and thread error, but 
4.  It appears that diiferent kernel sizes are targeted toward different defects, with the the smallest 3x3 kernels being better for the _hole_ and larger 8x8 being better for the _objects_.  A logical next step would be to build a model that employs at least these two dimensions of filters.

INTRODUCTION

Tilda [click to see Tilda website](https://www.tildafabricusa.com) is a fabric brand founded by Norwegian designer Tone Finnanger in 1999, best known for whimsical naive characters and charming fabric design.

![Tilda products](./TILDA-defect-classific/images/introduction/living_room.jpg)



The Tilda image dataset for defect detection/classification in fabric presents an image classification on an industrial context (<https://www.kaggle.com/datasets/angelolmg/tilda-400-64x64-patches>). It is based on the public dataset by the Deutsche Forschungsgemeinschaft's Texture analysis group from the University of Freiburg (<https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html>) The original images were resized from 768x512 to 512x512, and then broken into 64x64 patches.

GOALS
1. Build a model that will detect defects images that contain defects. (2-class)
2. Build a model that will identify the defects in each image. (4-class)

I do not know in details the exact business use cases are.  However, at a high level, the 2-class model will be for catching any defective products from being shipped.  The 4-class model will be for identifying what part of the production or supply chain is causing the defects in the merchandise.

OVERVIEW OF DATA
1. The data is massively imbalanced, with 90.5% if the images being in the “good” class.
<p><img src='./TILDA-defect-classific/images/introduction/pop_dist_all_samples.png' width="800"><p>
<p><img src='./TILDA-defect-classific/images/introduction/pop_dist_defects.png' width="800"><p>

Below are the actual number of images for each class
| Class | Count |
| --- | --- |
| Good | 23,170 |
| Thread error | 620 |
| Oil spot | 636 |
| Hole | 337 |
| Objects | 837 |

Note that even for the defects, there is a strong imbalance. The number of hole samples is less than half of the Objects. This helps explain why the hole results are inferior to the others.

SAMPLES OF IMAGES FROM EACH CLASS:
Good
<p><img src='./TILDA-defect-classific/images/introduction/good.png' width="1000"><p>

Thread error (Defects)
<p><img src='./TILDA-defect-classific/images/introduction/thread_error.png' width="1000"><p>

Oil spot (Defects)
<p><img src='./TILDA-defect-classific/images/introduction/oil_spot.png' width="1000"><p>

Hole (Defects)
<p><img src='./TILDA-defect-classific/images/introduction/hole.png' width="1000"><p>

Objects (Defects)
<p><img src='./TILDA-defect-classific/images/introduction/objects.png' width="1000"><p>

From the above random samples, one can see the original classification may not have been perfect. For example,

1. Sample 086_patch4-3.png (bottom row, 4th from left) classified as _good_ appears to have a defect that looks like a thread error on it.
2. Sample 128_patch2-3.png (bottom row, middle) under _thread error_ does not contain any obvious defect to the naked eye.
3. Sample 034_patch3-1.png (top row, middle) classified as a _hole_ error does not appear to have a hole.

It is also interesting to note that the holes appear as bright spot, which leads me to believe that the contrast for all the samples have been reverse, or there is an illumination source from below the sample. However, that does not affect the modeling.

**MODEL OVERVIEW**

The images are 64x64 pixels. We employ a segmented approach. First we use a 2-class CNN model to detect the defects. Then we employ a 4-class model to identify the defects. Both models have a very similar architecture. In both models, I employed a CNN network of 3 convolutional blocks, followed by a flattening, fully connected layer, and a finally softmax output layer. The only difference between the two models are the final softmax layer.

1. We experimented with four (4) convolutional filter sizes: 3x3, 5x5, 7x7, and 9x9.  (Note previous we employed kernels with _even_ number sizes.  However, I have learned since then that even kernel sizes lead a problems when you desire the output to have the same size as the input. see [link](https://medium.com/geekculture/why-is-odd-sized-kernel-preferred-over-even-sized-kernel-a767e47b1d77)
2. Each convolutional block starts with with 8 or 16 conv. filters.
3. They are successively doubled to 8x2<sup>n</sup> or 16x2<sup>n</sup> in the following blocks.
4. After batch normalization, a 0.1 dropout layer is applied.
5. Each block ends with a 2x2 maxpooling layer.

The final output block is quite generic. The size of the fully connected layer is ¼ of the number of conv. filters in the last conv. block. The softmax layer becomes a signmoid layer when the number of classes = 2.

<p><img src='./TILDA-defect-classific/images/model_results/CNN_diagram.png' width="1000"><p>

ISSUES AND HOW THEY WERE ADDRESSED

**Filter size**: Small filters make the model train faster and can allow for more numerous filters to be applied, whereas large filters may capture correlations between features that are farther apart. I decided to employ only two sizes of 4x4 and 8x8 because 16x16 is already the half size of the output of the penultimate conv. block (Quick experiments, not shown here, confirmed it performed worse than the 4x4 and 8x8 filters).

**Monochromity**: The as-downloaded images contain the full complement of the 3 RGB colors, all having the same values. I reduced the dimensions from 64x64x3 to 64x64x1. However, this is tricky as some libraries automatically rescale the images, which is not desirable. At the end, I wrote my own function to perform this task.

**Dataset imbalance**: Each class is weighted by the inverse of its relative population during training. However, this is not sufficient. The problem is during stochastic gradient descent, the population of the minority class in each mini batch becomes so small that there is great variation from on mini batch to the next. For example, the hole class has only 337 images. The test set has 0.1 of the population (34 images). If we have 32 minibatches, we will end up with only 2 hole images for each mini batch.
The solution is we upsample/downsample each class to a reasonable number. In the defect detection model, we downsample the ‘good’ class to ~5120 samples, and upsample each of the defect classes to ~1240 samples (summing to ~5120 samples). This way, the sample sizes are balanced, and each minibatch has a good distribution of each population.

**Data augmentation**: It is a technique that involves generating additional images by modifying an original image, such as by flipping, rotating, reflecting, or changing the brightness and contrast of the image. Its use is widespread general image classification. However, in this case, I experimented with it and found the results to be markedly _worse_ than just simple upsampling. Therefore I did not employ this technique in this work.

**Maximizing Contrast**: This is one sub-example of data augmentation and may shed light into why data augmentation does not work in this use case. I performed a quick experiment in which I rescaled the pixel intensity to 0 ≤ i ≤ 1 instead of just dividing by 255, the performance became worse. I believe that is because the absolute intensity contains information about an object or feature’s height or depth which would be distorted or lost when the scales are changed. By the same token you would not apply image rotation to an auto-driving use case, since cars are never upside down, or image reflection to facial recognition since there is a difference between one’s regular and reflected image (e.g. hair partition).

I trained six model with the following parameters. They will be referred to as follows.
|     | **3x3 filters** | **5x5 filters** | **7x7 filters** | **9x9 filters** |
| --- | --- | --- | --- | --- |
| **8 filters + 16 filters + 32 filters (8,16,32)** | A-1 | B-1 | C-1 | D-1 |
| **16 filters + 32 filters + 64 filters (16,32,64)** | A-2 | B-2 | C-2 | D-2 |
|||||


**_RESULTS_**

**Defect DETECTION**

Receiver Operational Characteristics (ROC-AUC)
|     | **3x3 filters** | **5x5 filters** | **7x7 filters** | **9x9 filters** |
| --- | --- | --- | --- | --- |
| **8 filters + 16 filters + 32 filters (8,16,32)** | 93.1% | 91.8% | 91.8% | 88.4% |
| **16 filters + 32 filters + 64 filters (16,32,64)** | 91.5% | 92.4% | 89.2% | 89.8% |
|||||

The following shows the ROC for the model B-1:

<p><img src='./TILDA-defect-classific/images/model_results/roc_auc_B1.png' width="600"><p>

Precision
In a typical use case, there are two metrics that are important.
    1.  Precision for the “good” class, because we don’t want the “defected” textile to be mixed in with the “good” and shipped to the customers.
    2.  Recall for the "defect" class, because we want to know how many actual defects were flagged by the model (we want as high as possible).

The following is the precision confusion matrix for the model at 50% threshold with 8x8 (16,32, 64) filters. The threshold is set to 0.5. As one can see, the good predictions are 95% correct and the defect predictions are 92% correct.

Precision for the “good” class at 50% threshold.

|     | **3x3 filters** | **5x5 filters** | **7x7 filters** | **9x9 filters** |
| --- | --- | --- | --- | --- |
| **8 filters + 16 filters + 32 filters (8,16,32)** | 95.8% | 95.0%<sup>*<sup> | 95.0%<sup>*<sup> | 95.0%<sup>*<sup> |
| **16 filters + 32 filters + 64 filters (16,32,64)** | 94.5% | 95.4% | 96.1% | 94.4% |
|||||

<sup>*</sup> These values appear the same only by co-incidence.   They are different after the 3<sup>rd</sup> significant figure.

Recall for the "defect" class at 50% threshold.

|     | **3x3 filters** | **5x5 filters** | **7x7 filters** | **9x9 filters** |
| --- | --- | --- | --- | --- |
| **8 filters + 16 filters + 32 filters (8,16,32)** | 58.4% | 54.7% | 61.7% | 44.0% | 
| **16 filters + 32 filters + 64 filters (16,32,64)** |44.9% | 50.6% | 50.2% | 50.6% | 
|||||

**2-class defect detection summary**

Model A-2 has the largest ROC-AUC, which is the most-used metric to evaluation the effectiveness of a 2-class model.  However, in our case, with the threshold set at 50% probability, Model B-1 has a far superior Recall for the defect class (61.7% vs. 50.6%).  It is possible that the relative performance of the two models will flip, given model A-2 has a higher ROC_AUC than B-1.  However, the threshold depends on the business use case, and we are not privy to this information from the data set description. For this work we will assume 50% is the correct threshold.

Overall, the results are satisfactory given the simplicity of the model. The ROC-AUC of 91.8% and 92.4% are both quite impressive, especially given the large number of images that were hard to classified even for the human eye.  The Precision for the _good_ class of 96.1% is impressive, and the Recall for the _defect_ class of 61.7% is an improvement over the previous iteration of the models, even if there is room for improvement.

**Defect IDENTIFICATION**

Identification of the defects can be an important step to improving of the manufacturing process. From a modeling perspective, it presents a much more challenging problem. For one thing, the number of samples are small (see first table above _actual number of images for each class_). Further, there are 4 possibilities of defects, so the probability of getting it right is only about 25%. Note that 25% is the approximate baseline of the model: it must perform significantly above it for model to be called effective.

A close inspection of the images of defects in the section earlier will reveal that many classifications are almost arbitrary to the human eye. The oil spots appear to be just some shades on the textile. The holes do not always appear to be holes. The objects and thread error classes are also similar looking. Another issue is the imbalance of the data set. There are very few examples of the “hole” cases. Thus it is not surprising the models perform the worst for the class.

We will use the lower-case letters to refer to these 4-class models here to identification avoid confusion with the 2-class models for detect detection.

|     | **3x3 filters** | **5x5 filters** | **7x7 filters** | **9x9 filters** |
| --- | --- | --- | --- | --- |
| **8 filters + 16 filters + 32 filters (8,16,32)** | a-1 | b-1 | c-1 | d-1 |
| **16 filters + 32 filters + 64 filters (16,32,64)** | a-2 | b-2 | c-2 | d-2 | 
|||||

Below is the overall accuracy of the model. As one can see, the overall accuracies of the A1 model linger near the 50-60% levels.  While they are well above the baseline of ~25%, theprobable because of its simplicity. As for the most complex model B2, again it does not perform as well as the two models of medium complexity.

We use the overall accuracy as the metric to select the best model. The results are summarized below.

Total accuracy

|     | **3x3 filters** | **5x5 filters** | **7x7 filters** | **9x9 filters** |
| --- | --- | --- | --- | --- |
| **8 filters + 16 filters + 32 filters (8,16,32)** | 58.0% | 61.7% | 59.3% | 60.9% |
| **16 filters + 32 filters + 64 filters (16,32,64)** | 58.4?% | 59.7% | 63.0% | 52.3% |
|||||


We can see the model a-2 is the best one, marginally better than b-1. The a-1 model is barely above the 25% baseline, probably because the this is now a 4-class problem with too much complexity for it. The most complex model, b-4, does not perform well probably because of overfitting.

**Model b-1 (7 x 7 filters, 8 filters in the first layer)**

The following are the relevant confusion matrix for model B-2. Recall measures how many of the actual defects were flagged by the model.  It is the best metric here because _a-priori_ we are not aware of any preference for a class, like we know the defect class is highly undesirable for the 2-class model and needs to be screened out.

Here one can see the model performs especially well for the "thread error" and “objects” classes and poorly for the “hole” class. These are the most and least populous class in the data set, respectively, so this is all expected.

<figure>
<p><img src='./TILDA-defect-classific/images/model_results/3-5-7-9_kernels/4-class_7x7-16_count_conf_mat.png' width="600"><p>
    <figcaption>Confusion matrix for _count_ of each class (holdout)</figcaption>
</figure>


<figure>
    <img src='./TILDA-defect-classific/images/model_results/3-5-7-9_kernels/4-class_7x7_16_recall_conf_mat.png' width="600">
    <figcaption>Confusion matrix for _recall_ of each class (holdout)</figcaption>
</figure>

**Improvements opportunities**

1. There is strong evidence from the (4-class) defect identification model results that different filter sizes can target different defect types. The new model should have 2 or more filter sizes instead of one.

2. The dearth of defect class samples is clearly a direct cause of inferior results (e.g. holes). One way to just artificially increase the weights of these samples so the model will pay more attention to them. However, this will be done at the expense of accuracy for the other classes. Another possible avenue to explore is to train a Generative Adversarial Network (GAN) to generate more defect samples. However, this work will require probably a lot more effort than building these two models so far, so the business case must justify it to proceed.



**_SHAPLEY value analysis_**

[_Shapley Values_](https://www.investopedia.com/terms/s/shapley-value.asp#:~:text=Essentially%2C%20the%20Shapley%20value%20is,or%20less%20than%20the%20others.)  are derives from the _marginal contribution_ of each feature to a prediction, after applying all other features. In game theory, Shapley values help determine how much each player in a collaborative game has contributed to the total payout. For a machine learning model, each feature is considered a "player". The Shapley value for a feature represents the average magnitude of that feature's contribution across all possible combinations of features. Specifically, Shapley values are calculated by comparing a model's predictions with and without a particular feature present. This is done iteratively for each feature and each sample in the dataset. By assigning each feature an importance value for every prediction, SHAP values provide a local, consistent explanation of how the model behaves. They reveal which features have the most impact on a specific prediction, whether positively or negatively. This is valuable for understanding the reasoning behind complex machine learning models such as [_deep neural networks_](https://www.kdnuggets.com/2023/08/shap-values-model-interpretability-machine-learning.html)

In a CNN model, the basic feature is the pixel. We employ the [shap](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) library [DeepExplainer](https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/Front%20Page%20DeepExplainer%20MNIST%20Example.html) function to visualize each pixel contribution to the prediction. Because the shap library is a little out of date, I had to write my own code display the results.

DEFECT DETECTION (2-CLASS)

**<span style="color:red">TRUE POSITIVE (Actual = “defect”, Predicted = “defect”)**</span>.

    The image below shows the negative (blue) and positive (red) shapley values for each pixel of the image. The image was correctly classified by the model as “defect” (100% probability). One can see that the anomalous object on the lower right is drawing the positive attention (i.e. "there is a defect") of the model.  There is no feature that generated negative (i.e. "not defect") for the model. 
<p><img src='./TILDA-defect-classific/images/shap_plots/2-class/3x3_8-16-32_model/True_positives/GS_2cl_reload_shapval_[3]_8_img2494.png' width="600" style="border: 5px solid red;"><p>

    Another defect example is following, in which the model correctly located the anomalous feature on the image. There are scattered blue pixels in the image as well, but not nearly as numerous as red pixels to swing the prediction to negative.
 <p><img src='./TILDA-defect-classific/images/shap_plots/2-class/3x3_8-16-32_model/True_positives/GS_2cl_reload_shapval_[3]_8_img2546.png ' width="600" style="border: 5px solid red;"><p>

**<span style="color:orange">TRUE NEGATIVE (Actual = “good”, predicted = “good”)**</span>.

For the correctly predicted "good" images, they typically fall into two types.
    
    1. The images are very clean, and the model did not find any features of interest.  The following is an example.
<p><img src='./TILDA-defect-classific/images/shap_plots/2-class/3x3_8-16-32_model/True_negatives/GS_2cl_reload_shapval_[3]_8_img2495.png' width="600" style="border: 5px solid orange;"><p>

    2.  The images are very noisy or "dirty."  It has all kinds of features but of the types that indicate a defect and ones that indicate a clean sample.  They features are about equal in numbers, so they cancel each other.  Here is an example.
<p><img src='./TILDA-defect-classific/images/shap_plots/2-class/3x3_8-16-32_model/True_negatives/GS_2cl_reload_shapval_[3]_8_img0846.png' width="600" style="border: 5px solid orange;"><p>

**<span style="color:blue">FALSE POSITIVE (Actual = “Good”, Predicted = “Defect”)**</span>.
Like the True negatives, the images fell into two types.
    
    1.  The images are very noisy or "dirty."  However, in this case, the model mistook the noise for actual defects and misclassified them as such.  The following is an example.
<p><img src='./TILDA-defect-classific/images/shap_plots/2-class/3x3_8-16-32_model/False_positives/GS_2cl_reload_shapval_[3]_8_img1396.png' width="600" style="border: 5px solid blue;"><p>

    2.  The images are overall very clean but with very light features that are mistaken for a defects.  Below is an example. 
<p><img src='./TILDA-defect-classific/images/shap_plots/2-class/3x3_8-16-32_model/False_positives/GS_2cl_reload_shapval_[3]_8_img0141.png' width="600" style="border: 5px solid blue;"><p>

**<span style="color:green">FALSE NEGATIVE (Actual = “defect”, Predicted = “good”)**</span>.
This is probably the most important error for a business, because it is actually missing a sample that is defected.   There are two general types:
    1.  The defect feature is so faint it was difficult to catch.  Here is an example.
    <p><img src='./TILDA-defect-classific/images/shap_plots/2-class/3x3_8-16-32_model/False_negatives/GS_2cl_reload_shapval_[3]_8_img1106.png' width="600"  style="border: 5px solid green;"><p>

    2.  This is my opinion, but there are some samples that are mislabeled as defects. Below is an example. I challenge anyone to tell me this is a defect.  The data set has many of these.  The model, not surprising, only found a few pixels that would indicate there is a defect, and classified it as _good_.
    
<p><img src='./TILDA-defect-classific/images/shap_plots/2-class/3x3_8-16-32_model/False_negatives/GS_2cl_reload_shapval_[3]_8_img1916.png' width="600"  style="border: 5px solid green;"><p>

DEFECT DETECTION (4-Class)

When viewing the 4-class shapley plots, it is especially important to pay attention to the negative (blue) shapley values because they represent _counter-evidence_ of an image being in that class. Often an image is red on one side, but also blue or even blue on the other, thus the positive evidence all gets canceled.

<span style="color:red">**1. OBJECTS.**</span>
    
_CORRECT PREDICTIONS_
    Here are two examples in which there are strong red in classes other than the correct “object” classes, but those were overwhelmed by the strong negative values. Thus this makes the object class as the most likely prediction.  Note in both cases there is also a lot of signal for the class _thread errors_, indicating the model having a hard distinguishing them (see Recall confusion matrix above).
    <p><img src='./TILDA-defect-classific/images/shap_plots/4-class/objects/correct/GS_4cl_reload_shapval_7_16_img0038.png' width="400"  style="border: 5px solid red;"> <img src='./TILDA-defect-classific/images/shap_plots/4-class/objects/correct/GS_4cl_reload_shapval_7_16_img0175.png' width="400" style="border: 5px solid red;"><p>


_INCORRECT PREDICTIONS_
     <p><img src='./TILDA-defect-classific/images/shap_plots/4-class/objects/incorrect/GS_4cl_reload_shapval_7_16_img0206.png' width="400" style="border: 5px solid red;"> <img src='./TILDA-defect-classific/images/shap_plots/4-class/objects/incorrect/GS_4cl_reload_shapval_7_16_img0162.png' width="400" style="border: 5px solid red;"><p>

<span style="color:orange">**2. HOLE.**</span>

Hole is a challenging category for the model.  One reason is the lack of samples.  There are onl 337 samples of _hole_ defects compared to 620 for the next numerous category of _thread error_.  Another possibility is mislabeling.  I will demostrate below.

_CORRECT PREDICTIONS_:
    These images are classified as hole mostly based only the positive shapley values of the “hole” category.  <p><img src='./TILDA-defect-classific/images/shap_plots/4-class/hole/correct/GS_4cl_reload_shapval_7_16_img0089.png' width="400" style="border: 5px solid orange;"> <img src='./TILDA-defect-classific/images/shap_plots/4-class/hole/correct/GS_4cl_reload_shapval_7_16_img0020.png' width="400" style="border: 5px solid orange;"><p>

_INORRECT PREDICTIONS_:
    The following two images both do not appear like "holes."  This left image received a "thread error" prediction, which appears to be correct.  The right image received an "object" prediction from the model.  I am not able to see any defects on the sample, so perhaps it should have been labeled as a _good_ sample. <p><img src='./TILDA-defect-classific/images/shap_plots/4-class/hole/incorrect/GS_4cl_reload_shapval_7_16_img0016.png' width="400" style="border: 5px solid orange;"> <img src='./TILDA-defect-classific/images/shap_plots/4-class/hole/incorrect/GS_4cl_reload_shapval_7_16_img0022.png' width="400" style="border: 5px solid orange;"><p>

<span style="color:blue">**3. OIL SPOT.**</span>
    
_CORRECT PREDICTIONS_:
    Here is an image the can appear as an _oil spot_ or a _thread error_ to the human eye. However, the model correctly surmised there is more evidence for _oil spot_ and more counter-evidence against _thread error,_ and predicted _oil spot._ <p><img src='./TILDA-defect-classific/images/shap_plots/4-class/oil_spot/correct/GS_4cl_reload_shapval_4_16_img170.png' width="400" style="border: 5px solid blue;"> <img src='./TILDA-defect-classific/images/shap_plots/4-class/oil_spot/correct/GS_4cl_reload_shapval_7_16_img0045.png' width="400" style="border: 5px solid blue;"><p>

_INORRECT PREDICTIONS_:
    The _oil spot_ iamges that foiled the models tend to look have large blotches that look like objects.  <p><img src='./TILDA-defect-classific/images/shap_plots/4-class/oil_spot/incorrect/GS_4cl_reload_shapval_7_16_img0225.png' width="400" style="border: 5px solid blue;"> <img src='./TILDA-defect-classific/images/shap_plots/4-class/oil_spot/incorrect/GS_4cl_reload_shapval_7_16_img0189.png' width="400" style="border: 5px solid blue;"><p>

<span style="color:green">**4. THREAD ERROR**</span>.
    
_CORRECT PREDICTIONS_:
    The thread error defects tend to be bright objects that are long in shape. <p><img src='./TILDA-defect-classific/images/shap_plots/4-class/thread_error/correct/GS_4cl_reload_shapval_7_16_img0004.png' width="400" style="border: 5px solid green;"> <img src='./TILDA-defect-classific/images/shap_plots/4-class/thread_error/correct/GS_4cl_reload_shapval_7_16_img0210.png' width="400" style="border: 5px solid green;"><p>

_INORRECT PREDICTIONS_:
     The false thread erro predictions tend to _not_ long in shape. For example, the image on the right did not a have clearly defined bright spot, and so it was predicted as an _oil spot_.  The image on the left has a round bright or oblong bright spot, so it was predicted to be of _objects_ class.  It is entirely possible for these two images to have been mislabeled. <p><img src='./TILDA-defect-classific/images/shap_plots/4-class/thread_error/incorrect/GS_4cl_reload_shapval_7_16_img0048.png' width="400" style="border: 5px solid green;"><img src='./TILDA-defect-classific/images/shap_plots/4-class/thread_error/incorrect/GS_4cl_reload_shapval_7_16_img0043.png' width="400" style="border: 5px solid green;"><p>


**APPENDIX: CNN FILTERS  (_For curiosity only, not much utility in this use case_)**

It is generally interesting if not educational to examine the filters to see what kinds of generalized features the CNN is paying attention to. For example, in the automobile classification model (<https://www.analyticsvidhya.com/blog/2021/06/beginner-friendly-project-cat-and-dog-classification-using-cnn/>) we can see the the low-level features are built up to form higher level featues that resemble car parts.

![Example filters for auto model](./TILDA-defect-classific/images/CNN_filters/filters_auto_model.png)

Unfortunately, in this use case, the generalized features are not so obvious. The likely explanation is that the defects that are irregular in shape and sizes, so the fundamental building blocks more difficult to visualize the break-down. Nevertheless, **for completeness**, we show some selected examples of the features of the three convolutional blocks of the detection and identification models.

<span style="color:red">**CNN FILTERS FOR _2-CLASS_ DEFECT DETECTION MODEL:**

<span style="color:red">1<sup>st</sup> layer filters**
<p><img src='./TILDA-defect-classific/images/CNN_filters/2-class/2class_layer0filters.png' width="800"  alt="your-image-description"  style="border: 5px solid RED;"><p>

<span style="color:red">**2nd block filters (random sample of 128)</span>.**
<span style="color:red"><p><img src='./TILDA-defect-classific/images/CNN_filters/2-class/2class_layer4filters.png' width="800" style="border: 5px solid red;"><p>

<span style="color:red">**3<sup>rd</sup> block filters (random sample of 128)**
<p><img src='./TILDA-defect-classific/images/CNN_filters/2-class/2class_layer8filters.png' width="800" style="border: 5px solid RED;"><p>

<span style="color:green">**CNN FILTERS FOR _4-CLASS_ DEFECT IDENTIFICATION MODEL:**

<span style="color:green">**1<sup>st</sup> layer filters**
<p><img src='./TILDA-defect-classific/images/CNN_filters/4-class/4class_layer0filters.png' width="800" style="border: 5px solid green;"><p>

<span style="color:green">**2<sup>nd</sup> block filters (random sample of 128)****
<p><img src='./TILDA-defect-classific/images/CNN_filters/4-class/4class_layer4filters.png' width="800" style="border: 5px solid green;"><p>

<span style="color:green">**3<sup>rd</sup> block filters (random sample of 128)**</span>.
<p><img src='./TILDA-defect-classific/images/CNN_filters/4-class/4class_layer8filters.png' width="800" style="border: 5px solid green;"><p>