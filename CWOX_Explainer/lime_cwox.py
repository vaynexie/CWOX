import os, json
import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import check_random_state
from skimage.segmentation import mark_boundaries, felzenszwalb, slic, quickshift
from PIL import Image
import scipy as sp
from tqdm.auto import tqdm
from base_explainer.attribution.lime import get_image,get_input_tensors,get_pil_transform,get_preprocess_transform
from base_explainer.attribution.lime import batch_predict,has_arg,BaseWrapper,SegmentationAlgorithm,ImageExplanation,LimeBase,LimeImageExplainer
from base_explainer.attribution.lime import output_exp_final,output_exp_final_final,times_support,get_image_and_mask1,sal_map_to_boundry

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

def inverse_trans(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i, (m, s) in enumerate(zip(mean, std)):
        img[:, i] *= s
        img[:, i] += m
    return img

def lime_cwox(img,model,cluster_use_final,kernel_size1=5,number_sample1=3000):

#kernel_size1:kernel_size for segmentation algorithm (larger kernel size, larger super pixels)
#number_sample1: number of samples used for fitting regression function
    img = img.clone()
    img = inverse_trans(img)
    img = transforms.ToPILImage()(img[0])
    explainer = LimeImageExplainer()
    sal_dict={}
    random_state = check_random_state(None)
    random_seed = random_state.randint(0, high=1000)
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=kernel_size1,
                                        max_dist=10, ratio=1,random_seed=random_seed)
    segments = segmentation_fn(np.array(pill_transf(img)))
    for i in range(len(cluster_use_final)):
        class_use=cluster_use_final[i]
        class_not_use=[]
        for j in [x for x in range(len(cluster_use_final)) if x != i]:
            class_not_use.append(cluster_use_final[j])
        explanation = explainer.explain_instance(np.array(pill_transf(img)),model, 
                                             batch_predict,segments,
                                             pos_labels=class_use,
                                             neg_labels=class_not_use,
                                             hide_color=0, 
                                             num_samples=number_sample1) 
        key_list=list(explanation.local_exp.keys())
        cwox1_neg_list=[s for s in key_list if "cwox1_neg" in s]
        exp_pos=explanation.local_exp['cwox1_pos']
        exp_neg=[]
        for kkk in cwox1_neg_list:
            exp_neg.append(explanation.local_exp[kkk])
        if len(exp_neg)>0:
            exp_final_final=output_exp_final_final(exp_pos,exp_neg)
            stage1_result=exp_final_final
        if len(exp_neg)==0:
            stage1_result=exp_pos
        sal_dict['cluster'+str(i+1)]=stage1_result
        for pos_class in class_use:
            exp_pos=explanation.local_exp['cwox'+str(pos_class)]
            class_not_use1=[]
            for j in class_use:
                if j != pos_class:
                    class_not_use1.append(j) 
            neg_list=[]
            for neg_class in class_not_use1:
                neg_list.append(explanation.local_exp['cwox'+str(neg_class)])
            if len(neg_list)>0:
                stage2_result=output_exp_final_final(exp_pos,neg_list)
                #cwox2_support=times_support(stage1_result,stage2_result,delta)
                sal_dict['cluster'+str(i+1)+'_'+str(pos_class)]=stage2_result
            if len(neg_list)==0:
                stage2_result=exp_pos
                sal_dict['cluster'+str(i+1)+'_'+str(pos_class)]=stage2_result
    seg=explanation.segments
    sal_map_dict={}
    for sal in sal_dict:
        saliency_map=np.zeros((224,224))
        explain=sal_dict[sal]
        explain_dict={}
        for i in explain:
            explain_dict[i[0]]=i[1]
        for i in  range(len(seg)):
            for j in range(len(seg[i])):
                if seg[i][j] in explain_dict:
                    if explain_dict[seg[i][j]]>0:
                        saliency_map[i][j]=explain_dict[seg[i][j]]
        saliency_map=saliency_map/saliency_map.max()
        sal_map_dict[sal]=saliency_map
    return sal_map_dict

#Testing Example:
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = models.resnet50(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# model.to(device)
# img_path='eval_image//cello_guitar.jpg'
# cluster_use_final=[[486,889],[402,420,546]]
# delta=85
# kernel_size1=4
# number_sample1=2000
# sal_dict=lime_cwox(img_path,model,cluster_use_final,kernel_size1,number_sample1,delta)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = models.resnet50(pretrained=True)
# _ = model.train(False) # put model in evaluation mode
# model.to(device)
# img_path='eval_image//necklace.JPEG'
# cluster_use_final=[[635, 826],[679], [902], [892]]
# delta=60
# kernel_size1=6
# number_sample1=3000
# sal_dict=lime_cwox(img_path,model,cluster_use_final,kernel_size1,number_sample1,delta)
