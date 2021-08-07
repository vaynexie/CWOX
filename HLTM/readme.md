
## **Forked from https://github.com/kmpoon/hlta-xai**

## HLTM for XAI

Hierarchical latent tree models (HLTM) [Chen et al., Latent tree models for hierarchical topic detection. AIJ 2017] are a class of probabilistic models developed to model word co-occurrence in documents.  Here, we use them to reveal what class labels are confusing to a classifier.

<p align="center">
 <img src="https://user-images.githubusercontent.com/69588181/123727213-e4931980-d8c3-11eb-80a1-04980a363b1e.png" height="150" width="350">

</p>

Rabbit and hare are confusing to humans. When presented with images of either of the classes, a person might find it difficult to decide whether to label them as rabbit or hare. Similarly, two classes are confusing to a neural network classifier if it, when processing images containing objects of either of the classes, has trouble determining which of them to use as the output class label, and hence give both high probabilities. 

Based on this intuition, we run the classifier on a set of examples, typically the training examples. For each example, we get a list of top class labels, which we regard as a short document. The confusing class labels detection problem is thereby transformed into the problem of identifying clusters of co-occurring words in documents. A HLTM was developed for solving the problem.

-----------------------------------------------------------------------------------------------------------------------
## HLTM Training Data Preparation

**/scripts/prepare_training_data.py:**
example python code to prepare HLTM training data by running ResNet50 on ImageNet Training Set. The output is a npy file with each row representing the co-occurance of class label (that example output npy file can be downloaded from: https://drive.google.com/file/d/17Um2yv3pPCArWs9HoNOhxHOQQfeBEl-B/view?usp=sharing).

It is built from the Top K prediction classes of ImageNet training set, where K=#classes with cumulative probability=0.95. One HLTM is built for one model: The first step is to run the model on the training set to get all output probabilities, and then select the Top K classes according to the probabilities (with the cases K=1 removed).


```
# Examples in the output .npy file
...
[486,889,202],
[402,420,546],
...
```

**/scripts/convert.py:**
code to convert the .npy file into sparse file format that can processed by the HLTM learning code. 



-----------------------------------------------------------------------------------------------------------------------
## Learning HLTM
#### The released package v1.0 HLTA-XAI.jar can be found in https://github.com/kmpoon/hlta-xai/releases/tag/v1.0

### Step 1. Building Models 


To build a hierarchical latent tree model (HLTM), run the following command:

```java -cp HLTA-XAI.jar xai.hlta.HLTA data_file output_name```

In the above, `data_file` is the name of the data file, `output_name` is the name of the output such that the resulting model will be `output_name.bif` and `HLTA-XAI.jar` is the jar library of the HLTA-XAI package.

For example:

```java -Xmx8G -cp HLTA-XAI.jar xai.hlta.HLTA test.sparse.txt.gz outmodel```

The resulting model will be named `outmodel.bif`.  The `-Xmx8G` specifies that 8GB of memory will be allocated for the Java runtime.

There are two options that may influence the structural learning.

1. `--struct-batch-size  <num>` indicates the sample size used for calculating the BIC score in the UD-test.  A larger size may lead to smaller label clusters in the model.  The number of samples used in XAI may be too large compared to the sample size that is used to derive the BIC score, hence there may be a need to specify a certain number.  The default value is 5000.  This means that when the BIC score is calculated, the BIC scores for batches of 5000 samples are calculated and then the average of the BIC scores is used in the UD-test.  It is suggested to try a range of values from 1,000 to 10,000.

2. `--struct-learn-size  <arg>` indicates the number of samples to be used in structural learning.  A larger number means more memory and CPU time will be needed.  If a number smaller than the original sample size is specified, a subset of samples will be randomly selected from the original data set.


### Step 2. Extracting Trees

After building the model, a topic tree displayed in a webpage can be built by the following command:

```java -cp HLTA-XAI.jar xai.hlta.ExtractTopicTree output_name model_file```

In the above, `output_name` is the name of the output tree, `output_name` is the name of the output such that the resulting topic tree can be opened from the file `output_name.html` and `HLTA-XAI.jar` is the jar library of the HLTA-XAI package.

-----------------------------------------------------------------------------------------------------------------------
## Example Results

We use ResNet50 and GoogleNet in the case of ImageNet Image Classification as examples. The whole output trees for the two models can be seen in:

<!-- TOC -->
- ResNet50 -- [Tree](https://hkust-huawei-xai.github.io/final_submit/resnet50), [Resutled JSON file](https://github.com/HKUST-HUAWEI-XAI/CWOX/blob/main/HLTM/result_json/ResNet50.json) ;  
- GoogleNet -- [Tree](https://hkust-huawei-xai.github.io/final_submit/googlenet), [Resutled JSON file](https://github.com/HKUST-HUAWEI-XAI/CWOX/blob/main/HLTM/result_json/GoogleNet.json) .
<!-- TOC -->

For easy comprehension, a representative image of each class is included in the tree structures. Those images are not a part of the models. 

<p align="center">

 <img src="https://user-images.githubusercontent.com/69588181/123727847-e14c5d80-d8c4-11eb-8126-6eddaae6d588.png" height="250" width="700">
</p>
<div align="center">
 <b>Part of a hierarchical latent tree model (HLTM) obtained for ResNet50</b>
</div>






