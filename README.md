# Two-Stage Contrastive Whole-output Explanation (CWOX-2s)

## What Is CWOX-2s

![whole](https://github.com/vaynexie/CWOX/assets/55047915/45d6170b-5174-4a84-a079-5c6dda211def)
<div align="center",class='inBold'>
 <b>Figure 1</b>
</div>


 **CWOX-2s (Two-Stage Contrastive Whole-output Explanation)**  is a novel explanation framework where one can examine the evidence for competing classes, and thereby obtains contrastive explanations for Image Classification (see paper [add link when the paper is released to public] for details and citations).

---------

Requirement:

The main part of the codes is based PyTorch, please refer to the [**requirements.txt**](https://github.com/vaynexie/CWOX/blob/main/requirements.txt) for the detailed requirements of package version;

The building of hierarchical latent tree model (HLTM) in part A requires Java 11 and Scala 2.12.12.

----------

**In the following, we give the step-by-step tutorial for generating the CWOX-2s explanations.**


## [A. Label Confusion Clusters Idenification](https://github.com/xie-lin-li/CWOX/tree/main/HLTM)

CWOX-2s has a preprocessing step that partitions all class labels into confusion clusters with respect to the classifier to be explained. Classes in each of those clusters (e.g., cello, violin) are confusing to the classifier, and are often competing labels for the same object/region in the input image. CWOX-2s does so by analyzing the co-occurrence of labels in classification results and thereby building a hierarchical latent tree model (HLTM):

<p align="center">
 <img src="https://user-images.githubusercontent.com/69588181/120742320-3e086280-c529-11eb-9575-7e78593726fe.png" height="500" width="700">
</p>

<div align="center",class='inBold'>
 <b>Figure 2</b>
</div>

The codes for learning HLTMs are given in the sub-directory [HLTM](https://github.com/xie-lin-li/CWOX/tree/main/HLTM), along with the structures of the models obtained for [ResNet50](https://xie-lin-li.github.io/final_submit/resnet50) and [GoogleNet](https://xie-lin-li.github.io/final_submit/googlenet). The HLTM codes output a json file named **output_name_fullname.nodes.json**, which includes the learned hierarchical latent tree model. The json file is used in the following partition of label confusion clusters. The **ResNet50.json** in the example code shown below is renmaned from the **output_name_fullname.nodes.json**.

When interpreting the output of the classifier on a target image, CWOX-2s obtains a subtree for the top classes by removing from the HLTM all the irrelevant nodes. The top classes are partitioned into **Label Confusion Clusters** by cutting the subtree at a certain level, the default being the lowest level. This is how the two clusters in Figure 1 are obtained from the tree in Figure 2. 

The example codes for partitioning the top classes are given below:

```python
'''
Partition the Top-k labels into different clusters. 

User can select different cut_level to be used. In default, we apply the latent node in lowest level (cut_level=0) to divide the top classes into clusters.

The JSON files including the HLTM information we obtained for ImageNet Image Classification: ResNet50.json and GoogleNet.json can be found in 
https://github.com/xie-lin-li/CWOX/blob/main/HLTM/result_json or https://github.com/xie-lin-li/CWOX/tree/main/resources
'''
from CWOX.apply_hltm import *
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
## [B. CWOX-2s](https://github.com/xie-lin-li/CWOX/tree/main/CWOX)

CWOX-2s requires a base explainer, which can be any existing explanation methods, such as Grad-CAM, MWP, LIME and RISE, that yield nonnegative heatmaps. CWOX-2s first runs the base explainer on the confusion clusters (C_i’s) and the individual classes (c_ij’s), and then combines the base heatmaps to form contrastive heatmaps.

![image](https://user-images.githubusercontent.com/69588181/123724449-f1f9d500-d8be-11eb-98a4-c36953dd0ecc.png)


A score function is needed in order to produce a base heatmap for a class c. It is usually either the logit ![math](https://latex.codecogs.com/svg.image?z_c(\textbf{x})) of the class (for Grad-CAM and MWP) or the probability ![math](https://latex.codecogs.com/svg.image?p(c|\mathbf{x})) of the class (for RISE and LIME). For confusion clusters, the logit is replaced by the generalized logit ![math](https://latex.codecogs.com/svg.image?\inline&space;z_{\textbf{C}}=log\sum_{c\in\textbf{C}}{e}^{z_c}) and the probability is replaced by the probability of the cluster ![math](https://latex.codecogs.com/svg.image?\inline&space;p(\textbf{C}|\mathbf{x})=\sum_{c\in\textbf{C}}P(c|\mathbf{x})).


* [CWOX.IOX(algo)](https://github.com/xie-lin-li/CWOX/blob/main/CWOX/IOX.py): Produces a base heatmap using explanation method named algo. Currently, algo = “Grad-CAM”, “MWP”, “RISE”, or “LIME” are supported.

   * [CWOX.grad_cam_cwox](https://github.com/xie-lin-li/CWOX/blob/main/CWOX/grad_cam_cwox.py): Produces a base heatmap with Grad-CAM;
   * [CWOX.excitationbackprop_cwox](https://github.com/xie-lin-li/CWOX/blob/main/CWOX/excitationbackprop_cwox.py): Produces a base heatmap with MWP;
   * [CWOX.rise_cwox](https://github.com/xie-lin-li/CWOX/blob/main/CWOX/rise_cwox.py): Produces a base heatmap with RISE;
   * [CWOX.lime_cwox](https://github.com/xie-lin-li/CWOX/blob/main/CWOX/lime_cwox.py): Produces a base heatmap with LIME.

* [CWOX.CWOX](https://github.com/xie-lin-li/CWOX/blob/main/CWOX/CWOX.py): Produces contrastive heatmaps with CWOX.
* [CWOX.plt_wox.plot_cwox](https://github.com/xie-lin-li/CWOX/blob/main/CWOX/plt_wox.py): Visualize CWOX-2s results.


-----------------------------------------------------------------------------------------------------------------------
## C. Examples

**The following examples illustrate the use of CWOX-2s to explain the results of ResNet50 on one image. The complete code can be found at [CWOX_Example.ipynb](https://github.com/xie-lin-li/CWOX/blob/main/CWOX_Example.ipynb), and more testing images can be found in the sub-directory [eval_image](https://github.com/xie-lin-li/CWOX/tree/main/eval_image).**

```python
# Load Needed Package
import torch
from torchvision import datasets, models, transforms, utils
from PIL import Image
from CWOX.CWOX_2s import CWOX_2s
from CWOX.IOX import IOX
from CWOX.plt_wox import plot_cwox

# Load Image and Model
loader = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
img_path='eval_image//cello_guitar.jpg'
image = Image.open(img_path).convert('RGB')
image = loader(image).float()
image=torch.unsqueeze(image, 0)
model = models.resnet50(pretrained=True)
_ = model.train(False) # put model in evaluation mode
```

##### Example 1 Grad-CAM:
```python
from CWOX.grad_cam_cwox import grad_cam_cwox

IOX_cluster=IOX(grad_cam_cwox(model,layer='layer4'))
IOX_class=IOX(grad_cam_cwox(model,layer='layer3.5.relu'))

# Confusion Cluster information from HLTM (see the *Part A. Label Confusion Clusters Idenification* for how to obtain it)
cluster_use_final=[[486,889],[402,420,546]]
sal_dict=CWOX_2s(image,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=50,multiple_output=False)

# Make the plot for the CWOX-2s results
plot_cwox(sal_dict,image,cluster_use_final)
```
<img src="https://user-images.githubusercontent.com/69588181/116185069-7612cd80-a753-11eb-8bdc-fb987b4d978f.png" height="400" width="700">

##### Example 2 MWP:
```python
from CWOX.excitationbackprop_cwox import excitationbackprop_cwox

# Update a ResNet model to use :class:`EltwiseSum` for the skip connection.
from CWOX.base_explainer.attribution.excitation_backprop import update_resnet
model_update=update_resnet(model)
_ = model_update.train(False) # put model in evaluation mode

IOX_cluster=IOX(excitationbackprop_cwox(model_update,layer='layer4'))
IOX_class=IOX(excitationbackprop_cwox(model_update,layer='layer4.0.relu'))
    
# Confusion Cluster information from HLTM (see the *Part A. Label Confusion Clusters Idenification* for how to obtain it)
cluster_use_final=[[486,889],[402,420,546]]
sal_dict=CWOX_2s(image,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=60,multiple_output=False)

# Make the plot for the CWOX-2s results
plot_cwox(sal_dict,image,cluster_use_final)

```

<img src="https://user-images.githubusercontent.com/69588181/118123732-41776500-b427-11eb-8b4a-a504755dcd47.png" height="400" width="700">


##### Example 3 RISE:
```python
from CWOX.base_explainer.attribution.rise import rise
from CWOX.rise_cwox import rise_cwox

IOX_cluster=IOX(rise_cwox(model,N=5000,mask_probability=0.3,down_sample_size=15,gpu_batch=30))
IOX_class=IOX(rise(model,N=3000,mask_probability=0.14,down_sample_size=10,gpu_batch=30))
    
# Confusion Cluster information from HLTM (see the *Part A. Label Confusion Clusters Idenification* for how to obtain it)
cluster_use_final=[[486,889],[402,420,546]]
sal_dict=CWOX_2s(image,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=70,multiple_output=True)

# Make the plot for the CWOX-2s results
plot_cwox(sal_dict,image,cluster_use_final)

```
<img src="https://user-images.githubusercontent.com/69588181/117533288-8f066300-b01e-11eb-8374-92cb34886d5c.png" height="400" width="700">


##### Example 4 LIME:
```python
from CWOX.base_explainer.attribution.lime import lime
from CWOX.lime_cwox import lime_cwox

IOX_cluster=IOX(lime_cwox(model,kernel_size=4,number_sample=2000,gpu_batch=100))
IOX_class=IOX(lime(model,kernel_size=4,number_sample=2000,gpu_batch=100))

# Confusion Cluster information from HLTM (see the *Part A. Label Confusion Clusters Idenification* for how to obtain it)
cluster_use_final=[[486,889],[402,420,546]]
sal_dict=CWOX_2s(img_path,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=85,multiple_output=True)

# Make the plot for the CWOX-2s results
plot_cwox(sal_dict,image,cluster_use_final)

```
<img src="https://user-images.githubusercontent.com/69588181/117533306-a9404100-b01e-11eb-87a9-7178c1348562.png" height="400" width="700">


-----------------------------------------------------------------------------------------------------------------------
## [D. Evaluation on Contrastive Explanation](https://github.com/xie-lin-li/CWOX/tree/main/Evaluation)

The sub-directory [Evaluation](https://github.com/xie-lin-li/CWOX/tree/main/Evaluation) provides code to compute Evaluation Metrics for measuring Contrastive Faithfulness.


-----------------------------------------------------------------------------------------------------------------------
## [E. CWOX_Explainer (App)](https://github.com/xie-lin-li/CWOX/tree/main/CWOX_Explainer)

Application to perform the Contrastive Whole-out Explanation Process. Currently only ResNet50 and GoogleNet for ImageNet Image Classification are supported.

See the [README page](https://github.com/xie-lin-li/CWOX/blob/main/CWOX_Explainer/readme.md) in the sub-directory CWOX_Explainer for the guidelines to use the Application.

----------

# Enquiry

* Weiyan Xie (wxieai@cse.ust.hk) (The Hong Kong University of Science and Techonology)
