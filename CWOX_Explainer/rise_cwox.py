import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.autograd import Function, Variable
from torchvision import datasets, models, transforms, utils
import numpy as np
import os, sys, copy ; sys.path.append('..')
from base_explainer.attribution.rise import gkern, RISE, norm_data, cwox_step
from PIL import Image
from scipy.special import softmax
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.nn.functional import conv2d
cudnn.benchmark = True

preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Normalization for ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),])

read_tensor = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])


def one_v_one_RISE(class_use,class_not_use,p,N,p1,CL,mask):
    neg_sal_list=[] 
    pos=class_use
    pos_sal=cwox_step(pos,p,N,p1,CL,mask)
    if len(class_not_use)!=0:
        length=len(class_not_use)
        pos_sal=pos_sal*length
        for neg in class_not_use:         
            saliency =cwox_step(neg,p,N,p1,CL,mask)
            neg_sal_list.append(saliency)
        for saliency1 in neg_sal_list:
            pos_sal-=saliency1 
        return pos_sal
    if len(class_not_use)==0:
        return pos_sal


def rise_cwox(img,model,cluster_use_final,size_cluster = 12, p_cluster = 0.3, size_class = 10,p_class = 0.15):
#size_cluster/class: size of masked pixels for cluster/class
#p_cluster/class:probability of pixels to be masked for cluster/class

    # Load black box model for explanations
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.eval()
    model = model.cuda()
    for p in model.parameters():
        p.requires_grad = False
    

    #Generate Mask for cluster
    explainer = RISE(model, (224, 224))
    explainer.generate_masks(N=5000, s=size_cluster, p1=p_cluster)
    with torch.no_grad():
        sal,p,mask,N,p1,CL=explainer(img.cuda())

    #Generate Mask for class
    top_k=[]
    for i in cluster_use_final:
        top_k.extend(i)
    #if there is only class in all clusters, no need to generate class heatmap-->so just generate N=1
    if len(top_k)==len(cluster_use_final):
        explainer = RISE(model, (224, 224))
        explainer.generate_masks(N=1, s=1, p1=0.1)
    if len(top_k)!=len(cluster_use_final):
        explainer = RISE(model, (224, 224))
        explainer.generate_masks(N=3000, s=size_class, p1=p_class)
    with torch.no_grad():
        sal_1,p_1,mask_1,N_1,p1_1,CL_1=explainer(img.cuda())


    sal_dict={}
    for i in range(len(cluster_use_final)):
        class_use=cluster_use_final[i]
        class_not_use=[]
        for j in [x for x in range(len(cluster_use_final)) if x != i]:
            class_not_use.append(cluster_use_final[j])

        saliency1=one_v_one_RISE(class_use,class_not_use,p,N,p1,CL,mask)
        saliency1=np.clip(saliency1,a_min=0,a_max=None)
        saliency1=norm_data(saliency1)
        name1='cluster'+str(i+1)
        sal_dict[name1]=saliency1
        
        if len(cluster_use_final)==1:
            for k in range(len(class_use)):
                poss=[class_use[k]]
                class_use=np.asarray(class_use)
                mask_a = np.ones(len(class_use), bool)
                mask_a[k] = False
                negs= list(class_use[mask_a])
                class_use=list(class_use)
                
                saliency2=one_v_one_RISE(poss,negs,p_1,N_1,p1_1,CL_1,mask_1)
                saliency3=np.clip(saliency2,a_min=0,a_max=None)
                saliency3=norm_data(saliency3)
                name2='cluster'+str(i+1)+'_'+str(class_use[k])
                sal_dict[name2]=saliency3
                
        if len(cluster_use_final)>1 and len(class_use)>1:
            for k in range(len(class_use)):
                poss=[class_use[k]]
                class_use=np.asarray(class_use)
                mask_a = np.ones(len(class_use), bool)
                mask_a[k] = False
                negs= list(class_use[mask_a])
                class_use=list(class_use)
                saliency2=one_v_one_RISE(poss,negs,p_1,N_1,p1_1,CL_1,mask_1) 
                saliency3=np.clip(saliency2,a_min=0,a_max=None)
                saliency3=norm_data(saliency3)
                name2='cluster'+str(i+1)+'_'+str(class_use[k])
                sal_dict[name2]=saliency3
    return sal_dict


#Testing Example:
# model = models.resnet50(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image//cello_guitar.jpg'
# cluster_use_final=[[486,889],[402,420,546]]
# size_cluster=15
# p_cluster=0.3
# size_class=10
# p_class=0.14
# delta=60
# sal_dict=rise_cwox(img_path,model,cluster_use_final,size_cluster,p_cluster,size_class,p_class,delta)

# model = models.resnet50(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image//necklace.JPEG'
# cluster_use_final=[[635, 826],[679], [902], [892]]
# size_cluster=15
# p_cluster=0.3
# size_class=10
# p_class=0.14
# delta=50
# sal_dict=rise_cwox(img_path,model,cluster_use_final,size_cluster,p_cluster,size_class,p_class,delta)


# model = models.googlenet(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image//screwdriver.JPEG'
# cluster_use_final=[[784], [845]]
# size_cluster=10
# p_cluster=0.15
# size_class=None
# p_class=None
# delta=0
# sal_dict=rise_cwox(img_path,model,cluster_use_final,size_cluster,p_cluster,size_class,p_class,delta)
