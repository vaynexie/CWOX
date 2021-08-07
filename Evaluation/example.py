#-------------------Load the Package and Define needed Function-------------------
import torch,torchvision
import numpy as np 
from evaluation.utils import read_tensor
from evaluation.causal import CausalMetric, visual_evaluation
from evaluation.common import Explanation
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def normalize(x):
    x = x.clip(0) / x.max()
    x[x.isnan()] = 0
    return x

def load_example():
    img = read_tensor(img_path)
    img = img.to(device)
  	#User can aslo input other models as they want
    model_eval = torchvision.models.resnet50(pretrained = True)
    model_eval.to(device)
    model_eval.eval()
    return model_eval, img

#-------------------Load the Example and Explanation-------------------
'''
The image 'cello_guitar.jpg' can be found in eval_image: https://github.com/HKUST-HUAWEI-XAI/CWOX/tree/main/eval_image

ResNet50 Top 5 prediction classes for cello_guitar.jpg:
cello(0.839), acoustic-guitar (0.081), banjo (0.036), violin (0.021), electric-guitar (0.008)

Confusion Cluster information by HLTM: [[486,889],[402,420,546]] 
i.e. Cluster 1: cello, violin; Cluster 2: acoustic guitar, banjo, electric guitar.

cello.npy: CWOX results on cello_guitar example with Grad-CAM as base explainer
It is a dictionary with keys-486:cello, 889-violin,402-acoustic guitar,420-banjo,546-electric guitar; the value is the contrastive saliency heatmap for corresponding class
'''

img_path='cello_guitar.jpg'
sal_dict_Grad_CAM=np.load('cello.npy',allow_pickle=True).item()
model_eval, img11 = load_example()
# We evaluate on the cello heatmap now. Since we select the target class as 486-cello, the contrastive classs is 889-violin
check_ind=486
check_neg_ind=[889]
# Load the cello heatmap
saliency_used=sal_dict_Grad_CAM[check_ind]

#-------------------Define the metric-------------------
metric_type = ['CAUC', 'CDROP']
# delta (float): the delta for stoping the loop.
# tau_threshold(float): the threshold to penalize the CDROP score, where \tau=tau_threshold*Number of all pixels in the image
# smooth_len(int): For robustness, we smooth s(n_{\delta}+1) using a default sliding window of size 3.
# CDROP(H,m|x,c,C')=\frac{s(1)-s(n_{\delta}+1)}{log_{2}(1+max\{n_{\delta},\tau\}/\tau)}
metric = CausalMetric(model=model_eval,metric=metric_type, step = 16, batch_size = 8,delta = 0.5,tau_thres=0.05,smooth_len=3)
aaa=torch.tensor(saliency_used,dtype=torch.float64).to(device) 
saliency_single = normalize(aaa)
explain_contrast = Explanation(target = [[check_ind], check_neg_ind], saliency = saliency_single)

#------------------Run Evaluation with plotting and printing the result-------------------
from pylab import rcParams
rcParams['figure.figsize'] = 25, 20
show_vis = True
if show_vis:
    res, count = metric.single_run(img11, explain_contrast, visual_evaluation)
else:
    res, count = metric.single_run(img11, explain_contrast)

for i in res:
    print(str(i)+':'+str(res[i].cpu().item()))
plt.show()
