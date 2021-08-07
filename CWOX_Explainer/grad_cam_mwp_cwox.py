import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.autograd import Function, Variable
from torchvision import datasets, models, transforms, utils
import numpy as np
import os, sys, copy ; sys.path.append('..')
from base_explainer.attribution.gradient import gradient
from base_explainer.attribution.grad_cam import grad_cam
from base_explainer.attribution.excitation_backprop import excitation_backprop
from PIL import Image
from scipy.special import softmax
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.nn.functional import conv2d
cudnn.benchmark = True


loader = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])


def BP_logP(class_use,image1,layer_use,model,method):
    poss=class_use
    probs = model(image1).softmax(-1)
    grad = torch.zeros_like(probs)
    poss_p = probs[0, poss]
    grad[0, poss] = poss_p / poss_p.sum()
    if method=='Grad_CAM':
        saliency = grad_cam(model, image1, grad, saliency_layer=layer_use,resize = True)[0,0]
    elif method=='MWP':
        saliency = excitation_backprop(model, image1, grad, saliency_layer=layer_use,resize = True)[0,0]
    saliency=saliency.detach().cpu().numpy()
    return saliency

def one_v_one(class_use,class_not_use,image1,layer_use,model,method):
    neg_sal_list=[] 
    pos=class_use
    pos_sal=BP_logP(pos,image1,layer_use,model,method)
    if len(class_not_use)!=0:
        length=len(class_not_use)
        pos_sal=pos_sal*length
        for neg in class_not_use:         
            saliency = BP_logP(neg,image1,layer_use,model,method)
            neg_sal_list.append(saliency)
        for saliency1 in neg_sal_list:
            pos_sal-=saliency1 
        return pos_sal
    if len(class_not_use)==0:
        return pos_sal


def grad_cam_mwp_cwox(image1, model,cluster_use_final,layer_inter,layer_intra,method):
# method is either 'Grad_CAM' or 'MWP'

    sal_dict={}
    for i in range(len(cluster_use_final)):
        class_use=cluster_use_final[i]
        class_not_use=[]
        for j in [x for x in range(len(cluster_use_final)) if x != i]:
            class_not_use.append(cluster_use_final[j])
    
        saliency1=one_v_one(class_use,class_not_use,image1,layer_inter,model,method)
        saliency1=np.clip(saliency1,a_min=0,a_max=10000)
        name1='cluster'+str(i+1)
        sal_dict[name1]=saliency1

        if len(cluster_use_final)==1:
            for k in range(len(class_use)):
                poss=[class_use[k]]
                class_use=np.asarray(class_use)
                mask = np.ones(len(class_use), bool)
                mask[k] = False
                negs= list(class_use[mask])
                class_use=list(class_use)
                saliency2=one_v_one(poss,negs,image1,layer_intra,model,method)
                name2='cluster'+str(i+1)+'_'+str(class_use[k])
                sal_dict[name2]=np.clip(saliency2,a_min=0,a_max=10000)

        if len(cluster_use_final)>1 and len(class_use)>1:
            for k in range(len(class_use)):
                poss=[class_use[k]]
                class_use=np.asarray(class_use)
                mask = np.ones(len(class_use), bool)
                mask[k] = False
                negs= list(class_use[mask])
                class_use=list(class_use)
                saliency2=one_v_one(poss,negs,image1,layer_intra,model,method)  
                name2='cluster'+str(i+1)+'_'+str(class_use[k])
                sal_dict[name2]=np.clip(saliency2,a_min=0,a_max=10000)
    return sal_dict

def grad_cam_cwox(image1, model,cluster_use_final,layer_inter,layer_intra):
    return grad_cam_mwp_cwox(image1, model, cluster_use_final, layer_inter, layer_intra, "Grad_CAM")

def mwp_cwox(image1, model,cluster_use_final,layer_inter,layer_intra):
    return grad_cam_mwp_cwox(image1, model, cluster_use_final, layer_inter, layer_intra, "MWP")


    
# Testing Example:
# model = models.resnet50(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image/cello_guitar.jpg'
# layer_inter='layer4'
# layer_intra='layer3.5.relu'
# delta=50
# cluster_use_final=[[486,889],[402,420,546]]
# method='Grad_CAM'
# sal_dict=grad_cam_mwp_cwox(img_path,model,cluster_use_final,layer_inter,layer_intra,method,delta)
# print(sal_dict)

# model = models.resnet50(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image//cello_guitar.jpg'
# layer_inter='layer4'
# layer_intra='layer4.0.relu'
# delta=60
# cluster_use_final=[[486,889],[402,420,546]]
# method='MWP'
# sal_dict=grad_cam_mwp_cwox(img_path,model,cluster_use_final,layer_inter,layer_intra,method,delta)


# model = models.resnet50(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image//necklace.jPEG'
# layer_inter='layer4'
# layer_intra='layer4.1.relu'
# delta=60
# cluster_use_final=[[635, 826],[679], [902], [892]]
# method='Grad_CAM'
# sal_dict=grad_cam_mwp_cwox(img_path,model,cluster_use_final,layer_inter,layer_intra,method,delta)

# model = models.resnet50(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image//necklace.jPEG'
# layer_inter='layer4'
# layer_intra='layer4.0.relu'
# delta=60
# cluster_use_final=[[635, 826],[679], [902], [892]]
# method='MWP'
# sal_dict=grad_cam_mwp_cwox(img_path,model,cluster_use_final,layer_inter,layer_intra,method,delta)


# model = models.googlenet(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image//screwdriver.jPEG'
# layer_inter='inception5b.branch3'
# layer_intra=None
# delta=0
# cluster_use_final=[[784], [845]]
# method='Grad_CAM'
# sal_dict=grad_cam_mwp_cwox(img_path,model,cluster_use_final,layer_inter,layer_intra,method,delta)

# model = models.googlenet(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# img_path='eval_image//screwdriver.jPEG'
# layer_inter='inception5b.branch3'
# layer_intra=None
# delta=0
# cluster_use_final=[[784], [845]]
# method='MWP'
# sal_dict=grad_cam_mwp_cwox(img_path,model,cluster_use_final,layer_inter,layer_intra,method,delta)






