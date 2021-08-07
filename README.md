# Contrastive Whole-output Explanation (CWOX)

## What Is CWOX

![image](https://user-images.githubusercontent.com/69588181/123748855-5a5cac80-d8e7-11eb-8bf0-a6228e5236a3.png)

<div align="center",class='inBold'>
 <b>Figure 1</b>
</div>


 **CWOX (Contrastive Whole-output Explanation)**  is a novel explanation framework where one can examine the evidence for competing classes, and thereby obtains contrastive explanations for Image Classification (see paper [add link when the paper is released to public] for details and citations).
 
We provide Pytorch Implementation and Mindspore Implementation for CWOX:

<!-- TOC -->
- [Pytorch Implementation](https://github.com/vaynexie/CWOX)
- Mindspore Implementation - this branch

<!-- /TOC -->


## [A. Label Confusion Clusters Idenification](https://github.com/vaynexie/CWOX/tree/mindspore/HLTM)

CWOX has a preprocessing step that partitions all class labels into confusion clusters with respect to the classifier to be explained. Classes in each of those clusters (e.g., cello, violin) are confusing to the classifier, and are often competing labels for the same object/region in the input image. CWOX does so by analyzing the co-occurrence of labels in classification results and thereby building a hierarchical latent tree model (HLTM):

<p align="center">

 <img src="https://user-images.githubusercontent.com/69588181/120742320-3e086280-c529-11eb-9575-7e78593726fe.png" height="500" width="700">
</p>
<div align="center",class='inBold'>
 <b>Figure 2</b>
</div>

The codes for learning HLTMs are given in the sub-directory [HLTM](https://github.com/vaynexie/CWOX/tree/mindspore/HLTM), along with the structures of the models obtained for [ResNet50](https://vaynexie.github.io/final_submit/resnet50) and [GoogleNet](https://vaynexie.github.io/final_submit/googlenet).

When interpreting the output of the classifier on a target image, CWOX obtains a subtree for the top classes by removing from the HLTM all the irrelevant nodes. The top classes are partitioned into **Label Confusion Clusters** by cutting the subtree at a certain level, the default being the lowest level (the code for doing the parition is given in [mindspore.explainer.explanation.apply_hltm](https://github.com/vaynexie/mindspore_CWOX/blob/master/mindspore/explainer/explanation/_attribution/_CWOX/apply_hltm.py)). This is how the two clusters in Figure 1 are obtained from the tree in Figure 2. 

The example codes for using the **apply_hltm** to partition the top classes are given below:

```python
'''
Partition the Top-k labels into different clusters. 

User can select different cut_level to be used. In default, we apply the latent node in lowest level (cut_level=0) to divide the top classes into clusters.

The JSON files including the HLTM information we obtained for ImageNet Image Classification: ResNet50.json and GoogleNet.json can be found in 
https://github.com/vaynexie/CWOX/tree/mindspore/HLTM/result_json or https://github.com/vaynexie/CWOX/tree/mindspore/resources
'''

from mindspore.explainer.explanation import apply_hltm
clusterify_resnet50 = apply_hltm(cut_level=0,json_path="ResNet50.json")

'''
An Example: the top-5 prediction classes for eval_image/cello_guitar.jpg by ResNet50
486-cello 889-violin 402-acoustic guitar 420-banjo 546-electric guitar
The index for specified class can be checked in imagenet_class_index.json
'''
top_k=[486,889,402,420,546]
cluster_resnet50 = clusterify_resnet50.get_cluster(top_k)
print(cluster_resnet50)

'''
Output Results: [[486, 889], [402, 420, 546]],
which indicates that the top-5 classes are divided into two clusters suggested by the ResNet50 HLTM lowest-level latent node:
Cluster 1: cello, violin; Cluster 2: acoustic guitar, banjo, electric guitar.
'''
```

For the following discussions, we assume the top classes for an input x are partitioned into clusters: ![math](https://latex.codecogs.com/svg.image?\inline&space;C_1=\{c_{11},c_{12},\cdots\};C_2=\{c_{21},c_{22},\cdots\};\cdots).

-----------------------------------------------------------------------------------------------------------------------

## [B. Contrastive Whol-output Explanation (CWOX)](https://github.com/vaynexie/mindspore_CWOX/tree/master/mindspore/explainer/explanation/_attribution/_CWOX)

CWOX requires a base explainer, which can be any existing explanation methods, such as Grad-CAM, MWP, LIME and RISE, that yield nonnegative heatmaps. CWOX first runs the base explainer on the confusion clusters (C_i’s) and the individual classes (c_ij’s), and then combines the base heatmaps to form contrastive heatmaps.

![image](https://user-images.githubusercontent.com/69588181/123724449-f1f9d500-d8be-11eb-98a4-c36953dd0ecc.png)


A score function is needed in order to produce a base heatmap for a class c. It is usually either the logit ![math](https://latex.codecogs.com/svg.image?z_c(\textbf{x})) of the class (for Grad-CAM and MWP) or the probability ![math](https://latex.codecogs.com/svg.image?p(c|\mathbf{x})) of the class (for RISE and LIME). For confusion clusters, the logit is replaced by the generalized logit ![math](https://latex.codecogs.com/svg.image?\inline&space;z_{\textbf{C}}=log\sum_{c\in\textbf{C}}{e}^{z_c}) and the probability is replaced by the probability of the cluster ![math](https://latex.codecogs.com/svg.image?\inline&space;p(\textbf{C}|\mathbf{x})=\sum_{c\in\textbf{C}}P(c|\mathbf{x})).


* [mindspore.explainer.explanation.IOX(algo)](https://github.com/vaynexie/mindspore_CWOX/blob/master/mindspore/explainer/explanation/_attribution/_CWOX/IOX.py): Produces a base heatmap using explanation method named algo. Currently, algo = “Grad-CAM”, “MWP”, “RISE”, or “LIME” are supported.

   * [mindspore.explainer.explanation.grad_cam_cwox](https://github.com/vaynexie/mindspore_CWOX/blob/master/mindspore/explainer/explanation/_attribution/_CWOX/gradcam_cwox.py): Produces a base heatmap with Grad-CAM;
   * [mindspore.explainer.explanation.excitationbackprop_cwox](https://github.com/vaynexie/mindspore_CWOX/blob/master/mindspore/explainer/explanation/_attribution/_CWOX/excitationbackprop_cwox.py): Produces a base heatmap with MWP;
   * [mindspore.explainer.explanation.rise_cwox](https://github.com/vaynexie/mindspore_CWOX/blob/master/mindspore/explainer/explanation/_attribution/_CWOX/rise_cwox.py): Produces a base heatmap with RISE;
   * [mindspore.explainer.explanation.lime_cwox](https://github.com/vaynexie/mindspore_CWOX/blob/master/mindspore/explainer/explanation/_attribution/_CWOX/lime_cwox.py): Produces a base heatmap with LIME.

* [mindspore.explainer.explanation.CWOX](https://github.com/vaynexie/mindspore_CWOX/blob/master/mindspore/explainer/explanation/_attribution/_CWOX/CWOX.py): Produces contrastive heatmaps with CWOX.
* [mindspore.explainer.explanation.plot_cwox](https://github.com/vaynexie/mindspore_CWOX/blob/master/mindspore/explainer/explanation/_attribution/_CWOX/plt_wox.py): Visualize CWOX results.


## C. Examples

**The following examples illustrate the use of CWOX to explain the results of ResNet50 on one image. The complete code can be found at [CWOX_example.ipynb](https://github.com/vaynexie/CWOX/blob/mindspore/CWOX_example.ipynb), and more testing images can be found in the sub-directory [eval_image](https://github.com/vaynexie/CWOX/tree/mindspore/eval_image).**

```python
# Load CWOX explantion function and the plotting function
from mindspore.explainer.explanation import CWOX
from mindspore.explainer.explanation import IOX
from mindspore.explainer.explanation import plot_cwox

# Load example image and model
from mindspore.explainer.explanation._attribution._CWOX.imagenet_model.resnet50 import resnet_imagenet,read_preprocess
import mindspore as ms
from mindspore import Tensor
net=resnet_imagenet()
image_name='cello_guitar.jpg'
img_path='eval_image/'+str(image_name)
img_size = (224, 224)
image=read_preprocess(img_path,size=img_size)
inputs = Tensor(image, ms.float32)
```

#### Example 1 Grad-CAM:
```python
from mindspore.explainer.explanation import grad_cam_cwox

IOX_cluster=IOX(grad_cam_cwox(net,layer='layer4'))
IOX_class=IOX(grad_cam_cwox(net,layer='layer3'))

# Confusion Cluster information from HLTM (see the *Part A. Label Confusion Clusters Idenification* for how to obtain it)
cluster_use_final=[[486,889],[402,420,546]]
sal_dict=CWOX(inputs,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=50,multiple_output=False)

# Make the plot for the CWOX results
plot_cwox(sal_dict,image,cluster_use_final)
```
<img src="https://user-images.githubusercontent.com/69588181/116185069-7612cd80-a753-11eb-8bdc-fb987b4d978f.png" height="400" width="700">

#### Example 2 MWP:
```python
# Update a ResNet model to use :class:`EltwiseSum` for the skip connection.
from mindspore.explainer.explanation._attribution._backprop.excitationbackprop import update_resnet
net_update=update_resnet(net)

from mindspore.explainer.explanation import excitationbackprop_cwox

IOX_cluster=IOX(excitationbackprop_cwox(net_update,layer='layer4'))
IOX_class=IOX(excitationbackprop_cwox(net_update,layer='layer4.0.relu3'))

# Confusion Cluster information from HLTM (see the *Part A. Label Confusion Clusters Idenification* for how to obtain it)
cluster_use_final=[[486,889],[402,420,546]]
sal_dict=CWOX(inputs,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=60,multiple_output=False)

# Make the plot for the CWOX results
plot_cwox(sal_dict,image,cluster_use_final)
```

<img src="https://user-images.githubusercontent.com/69588181/116839821-486fcd80-ac06-11eb-9c78-3d2d4f48f1b0.png" height="400" width="700">



#### Example 3 RISE:
```python
from mindspore.explainer.explanation import RISE_quick
from mindspore.explainer.explanation import rise_cwox

IOX_cluster=IOX(rise_cwox(net,N=5000,mask_probability=0.3,down_sample_size=15,gpu_batch=20))
IOX_class=IOX(RISE_quick(net,N=3000,mask_probability=0.14,down_sample_size=10,gpu_batch=20))
    
# Confusion Cluster information from HLTM (see the *Part A. Label Confusion Clusters Idenification* for how to obtain it)
cluster_use_final=[[486,889],[402,420,546]]
sal_dict=CWOX(inputs,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=70,multiple_output=True)

# Make the plot for the CWOX results
plot_cwox(sal_dict,image,cluster_use_final)
```
<img src="https://user-images.githubusercontent.com/69588181/116184566-9a21df00-a752-11eb-9891-d67f9eae2976.png" height="400" width="700">


#### Example 4 LIME:
```python
from mindspore.explainer.explanation import LIME
from mindspore.explainer.explanation import lime_cwox

IOX_cluster=IOX(lime_cwox(net,kernel_size=4,number_sample=2000,gpu_batch=100))
IOX_class=IOX(LIME(net,kernel_size=4,number_sample=2000,gpu_batch=100))

# Confusion Cluster information from HLTM (see the *Part A. Label Confusion Clusters Idenification* for how to obtain it)
cluster_use_final=[[486,889],[402,420,546]]
sal_dict=CWOX(img_path,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=85,multiple_output=True)

# Make the plot for the CWOX results
plot_cwox(sal_dict,image,cluster_use_final)

```
<img src="https://user-images.githubusercontent.com/69588181/116187718-985b1a00-a758-11eb-9aa3-88f88184b635.png" height="400" width="700">



-----------------------------------------------------------------------------------------------------------------------



## [D. Evaluation on Contrastive Explanation](https://github.com/vaynexie/mindspore_CWOX/tree/master/mindspore/explainer/benchmark/_attribution)

The directory [mindspore_CWOX/mindspore/explainer/benchmark/_attribution](https://github.com/vaynexie/mindspore_CWOX/tree/master/mindspore/explainer/benchmark/_attribution) provides code to compute Evaluation Metrics for measuring Contrastive Faithfulness.


-----------------------------------------------------------------------------------------------------------------------



## [E. CWOX Explainer (App)](https://github.com/vaynexie/CWOX/tree/mindspore/CWOX_Explainer): 

Application to perform the Contrastive Whole-out Explanation Process. Currently only ResNet50 and GoogleNet for ImageNet Image Classification are supported.


See the [README page](https://github.com/vaynexie/CWOX/blob/mindspore/CWOX_Explainer/readme.md) in the sub-directory CWOX_Explainer for the guidelines to use the Application.
