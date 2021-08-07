# Experiment Dataset: A 10,000 split of ImageNet Validation Set
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torchvision import datasets, models, transforms, utils
from torch.utils.data.sampler import Sampler
from PIL import Image
import pickle as pkl
from pprint import pformat
from yapf.yapflib.yapf_api import FormatCode
from typing import Callable, Union, List, Dict, Tuple
import os, sys, copy ; sys.path.append('..')
from os import path
reseed = lambda: np.random.seed(seed=1) ; ms = torch.manual_seed(1) # for reproducibility
reseed()
from evaluation.causal import CausalMetric, visual_evaluation
from evaluation.common import Explanation
from torchray.attribution.gradient import gradient
from torchray.attribution.grad_cam import grad_cam
from scipy.special import softmax
import json 


def model_return_eval(model_input):
    if model_input=='ResNet50':
        model = models.resnet50(pretrained=True)   
    elif model_input=='GoogleNet':
        model = models.googlenet(pretrained=True) 
    return model

class ClusterTree:
    def __init__(self, load_path="./node_dict.pkl"):
        with open(load_path, "rb") as f:
            self.tree = pkl.load(f)
        # find all pathes in tree to leaf
        def find_all_paths(x, prefix=["root"]):
            if isinstance(x, list):  # height 1 node
                return [[*prefix, val] for val in x]
            else:  # others
                return sum(
                    [find_all_paths(vals, prefix + [key]) for key, vals in x.items()],
                    [],)
        self.paths = {path[-1]: path for path in find_all_paths(self.tree)}
    def get_cluster(self, indices, cut_depth=4):
        cluster = {i: self.paths[i][cut_depth] for i in indices}
        indices_map = {c:[] for c in cluster.values()}
        for i, c in cluster.items():
            indices_map[c].append(i)
        return list(indices_map.values())


# Image preprocessing function
preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Normalization for ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),
    preprocess,
    lambda x: torch.unsqueeze(x, 0)])

loader = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])


def BP_logP(class_use,image1,layer_use,model,sign=0):
    poss=class_use
    probs = model(image1).softmax(-1)
    grad = torch.zeros_like(probs)
    poss_p = probs[0, poss]
    grad[0, poss] = poss_p / poss_p.sum()
    saliency = grad_cam(model, image1, grad, saliency_layer=layer_use,resize = True)[0,0]
    saliency=saliency.detach().cpu().numpy()
    return saliency

def one_v_one(class_use,class_not_use,image1,layer_use,model,sign):
    neg_sal_list=[] 
    pos=class_use
    pos_sal=BP_logP(pos,image1,layer_use,model,sign)
    if len(class_not_use)!=0:
        length=len(class_not_use)
        pos_sal=pos_sal*length
        #pos_sal=np.zeros_like(pos_sal)
        for neg in class_not_use:         
            saliency = BP_logP(neg,image1,layer_use,model,sign)
            neg_sal_list.append(saliency)

        for saliency1 in neg_sal_list:
            pos_sal-=saliency1 
        return pos_sal     
    if len(class_not_use)==0:
        return pos_sal

def SCWOX(img_path,cluster_use_final,model,layer_inter,layer_intra):
    sal_dict={}
    sal_dict_display={}
    sal_dict2={}
    image = Image.open(img_path).convert('RGB')
    image = loader(image).float()
    image1=torch.unsqueeze(image, 0).to(device)
    top_k=[]
    for i in cluster_use_final:
        top_k.extend(i)
    k_t=len(top_k)
    sign=0
    if len(cluster_use_final)==1 and len(cluster_use_final[0])==k_t:sign=1
    for i in range(len(cluster_use_final)):
        class_use=cluster_use_final[i]
        class_not_use=[]
        for j in [x for x in range(len(cluster_use_final)) if x != i]:
            class_not_use.append(cluster_use_final[j])            
        saliency1=one_v_one(class_use,class_not_use,image1,layer_inter,model,sign)
        if len(cluster_use_final[i])==1 and len(cluster_use_final)!=k_t:saliency1=one_v_one(class_use,class_not_use,image1,layer_intra,model,sign)
        elif len(cluster_use_final[i])==1 and len(cluster_use_final)==k_t:saliency1=one_v_one(class_use,class_not_use,image1,layer_intra,model,sign)
        else:saliency1=one_v_one(class_use,class_not_use,image1,layer_inter,model,sign)
        name1='cluster'+str(i+1)
        sal_dict[name1]=saliency1
        sal_dict_display[name1]=np.clip(saliency1,a_min=0,a_max=10000)
        if len(cluster_use_final[i])==1:
            sal_dict2[cluster_use_final[i][0]]=np.clip(saliency1,a_min=0,a_max=10000)
        if len(cluster_use_final)==1:
            for k in range(len(class_use)):
                poss=[class_use[k]]
                class_use=np.asarray(class_use)
                mask = np.ones(len(class_use), bool)
                mask[k] = False
                negs= list(class_use[mask])
                class_use=list(class_use)
                saliency2=one_v_one(poss,negs,image1,layer_intra,model,sign)
                name2='cluster'+str(i+1)+'_'+str(class_use[k])   
                saliency_support =np.where(saliency1 >0, 1, 0)
                saliency3=saliency_support *saliency2
                sal_dict2[class_use[k]]=np.clip(saliency3,a_min=0,a_max=10000)
                sal_dict[name2]=saliency3
                sal_dict_display[name2]=np.clip(saliency3,a_min=0,a_max=10000)
        if len(cluster_use_final)>1 and len(class_use)>1:
            for k in range(len(class_use)):
                poss=[class_use[k]]
                class_use=np.asarray(class_use)
                mask = np.ones(len(class_use), bool)
                mask[k] = False
                negs= list(class_use[mask])
                class_use=list(class_use)
                saliency2=one_v_one(poss,negs,image1,layer_intra,model,sign)  
                saliency_support =np.where(saliency1 >0.0, 1, 0)
                saliency3=saliency_support *saliency2
                #saliency3/=(len(class_use)-1)
                sal_dict2[class_use[k]]=np.clip(saliency3,a_min=0,a_max=10000)
                name2='cluster'+str(i+1)+'_'+str(class_use[k])
                sal_dict[name2]=saliency3
                sal_dict_display[name2]=np.clip(saliency3,a_min=0,a_max=10000)
    sal_dict=sal_dict_display   
    sal_dict_key=list(sal_dict.keys())
    sal_dict_key_split=[]
    for i in sal_dict_key:
        sal_dict_key_split.append(i.split('_'))
    sal_dict_1={}
    for i in top_k:
        aaa=i
        for cluster_ind in range(len(cluster_use_final)):
            if i in cluster_use_final[cluster_ind]:
                temp_ind=cluster_ind      
        if len(cluster_use_final[temp_ind])>1 and len(cluster_use_final)!=k_t:sign=0      
        if len(cluster_use_final[temp_ind])==1 and len(cluster_use_final)!=k_t:sign=1
        if len(cluster_use_final[temp_ind])==1 and len(cluster_use_final)==k_t:sign=2
        temp_ind+=1
        if sign==0:
            for kkk in range(len(sal_dict_key_split)):
                if len(sal_dict_key_split[kkk])==1 and int(''.join(filter(str.isdigit, sal_dict_key_split[kkk][0])))==temp_ind:
                    sal_stage1=sal_dict[sal_dict_key[kkk]]               
                if len(sal_dict_key_split[kkk])==2  and sal_dict_key_split[kkk][1]==str(i):
                    sal_stage2=sal_dict[sal_dict_key[kkk]]
            sall=np.empty((224,224))
            sal_stage1=(sal_stage1-sal_stage1.min())/(sal_stage1.max()-sal_stage1.min())
            sal_stage2=(sal_stage2-sal_stage2.min())/(sal_stage2.max()-sal_stage2.min())
            for i in range(len(sal_stage1)):
                for j in range(len(sal_stage1)):
                    sall[i][j]=sal_stage1[i][j]/2+sal_stage2[i][j]
            salll_1=sall
            salll_1=np.clip(sall,a_min=0,a_max=10000)
            sal_dict_1[aaa]=salll_1  
        elif sign==1:    
            for kkk in range(len(sal_dict_key_split)):
                if len(sal_dict_key_split[kkk])==1 and int(''.join(filter(str.isdigit, sal_dict_key_split[kkk][0])))==temp_ind:
                    sal_stage1=sal_dict[sal_dict_key[kkk]]        
            sal_stage1=np.clip(sal_stage1,a_min=0,a_max=10000)
            sal_dict_1[aaa]=sal_stage1
        elif sign==2:
            sall=sal_dict['cluster'+str(list(top_k).index(aaa)+1)] 
            sall=np.clip(sall,a_min=0,a_max=10000)
            sal_dict_1[aaa]=sall
    return sal_dict2,sal_dict_1


def normalize(x):
    x = x.clip(0) / x.max()
    x[x.isnan()] = 0
    return x

def load_example():
    img = read_tensor(img_path)
    img = img.to(device)
    return img

def eval_result():
    torch.cuda.empty_cache() 
    res_list=[]
    res_n_list=[]
    count_list=[]
    for kk in range(len(top_k)):
        check_ind=top_k[kk]
        saliency_used=sal_dict2[check_ind]
        aaa=torch.tensor(saliency_used,dtype=torch.float64).to(device) 
        saliency_single = normalize(aaa)
        check_neg_ind=[]
        check_neg_ind1=[]
        for i in cluster:
            if check_ind in i:
                for j in i:
                    check_neg_ind1.append(j)  
                    if j!=check_ind:check_neg_ind.append(j)       
        count=0          
        if check_neg_ind!=[]:
            saliency_used=sal_dict2[check_ind]
            aaa=torch.tensor(saliency_used,dtype=torch.float64).to(device) 
            saliency_single = normalize(aaa)
            explain_contrast = Explanation(target = [[check_ind], check_neg_ind1], saliency = saliency_single , mode = 'contrast')
            res, res_n,count = metric.single_run(img11, explain_contrast)
            res_list.append(res[0])
            res_n_list.append(res_n[0])
        count_list.append(count)
    return res_list,res_n_list,count_list

##----------------------------------------------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_input='ResNet50'
model_eval=model_return_eval(model_input)
model_eval.to(device)
model_eval.eval()

clusterify = ClusterTree("node_dict_resnet50.pkl")
with open('C:\\Users\\vaynexie\\Desktop\\Evaluation\\labels_resnet50.json', "r") as read_file:aaa = json.load(read_file)
    
bbb={}
for i in aaa:
    if len(aaa[i])>1:
        length=len(aaa[i])
        result=clusterify.get_cluster(aaa[i])
        if len(result)!= len(aaa[i]):
            bbb[i]=result
image_list=list(bbb.keys())


for iii in range(len(image_list)):
    torch.cuda.empty_cache() 
    selected_image=image_list[iii]
    name_image=str(selected_image.split('.')[0].split('/')[1])+'_'+str(selected_image.split('.')[0].split('/')[2])
    img_path=selected_image
    cluster=bbb[image_list[iii]]
    layer_inter='layer4'
    layer_intra_list=['layer3.5.relu']
    top_k=[]
    for i in cluster:top_k.extend(i)
    if len(top_k)!=len(cluster):
        count_list_whole={}
        explanation_dict_whole={}
        result_dict_whole={}
        for layer_ind in range(len(layer_intra_list)):
            sal_dict2,sal_dict_1=SCWOX(img_path,cluster,model,layer_inter,layer_intra_list[layer_ind])
            model_eval, img11 = load_example()
            metric_type = ['AUC']
            metric = CausalMetric(model_eval, 'del', metric_type, step = 16, batch_size = 64,delta = 0.5)        
            res_list,res_n_list,count_list=eval_result()
            res_list1=[]
            for jj in range(len(res_list)):
                res_list1.append(res_list[jj].cpu().numpy())
            mean_res=np.mean(res_list1)
            count_list_whole[str(layer_intra_list[layer_ind])]=count_list
            explanation_dict_whole[str(layer_intra_list[layer_ind])]=sal_dict2
            result_dict_whole[str(layer_intra_list[layer_ind])]=res_list1
        np.save('count_list0126\\'+str(name_image)+'.npy',count_list_whole)
        np.save('explanation_dict0126\\'+str(name_image)+'.npy',explanation_dict_whole)
        np.save('result_dict0126\\'+str(name_image)+'_dict.npy',result_dict_whole)