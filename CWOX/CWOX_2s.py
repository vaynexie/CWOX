r"""
API to conduct Contrasitve Whole-output Explanation (CWOX) using Cluster Explainer and Class Explainer.
"""

import numpy as np

# Function to conduct the subtractions between class/cluster heatmaps
def one_v_one(class_use,class_not_use,sal_dict):
    neg_sal_list=[]
    pos=class_use
    pos_sal=sal_dict[str(pos)]
    if len(class_not_use)!=0:
        length=len(class_not_use)
        pos_sal=pos_sal*length
        for neg in class_not_use:  
            saliency = sal_dict[str(neg)]
            neg_sal_list.append(saliency)
        for saliency1 in neg_sal_list:
            pos_sal-=saliency1 
        return pos_sal
    if len(class_not_use)==0:
        return pos_sal

def CWOX_2s(image,cluster_list,cluster_method,class_method,delta=0,multiple_output=False):

    r"""
    Inputs:
        image (Tensor) - The input data to be explained, a 4D tensor of shape (N,C,H,W), only N=1 is supported now.
        cluster_list (List of List) - The cluster information that is used to make explanation. e.g. [[0,1],[2,3]]: 
        in this case class 0 and 1 are in the same cluster, and class 2 and 3 are in another cluster.
        cluster_method (Explanation Method) - The explanation method used for cluster heatmap.
        class_method (Explanation Method) - The explanation method used for class heatmap.
        delta (Float 0-100) - Parameter to control the distribution of class heatmap where only the pixels with saliency value in the cluster heatmap larger than min(saliency value)+delta*(max(saliency value)-min(saliency value)) will be considered, default is 0.
        multiple_output (Bool) - Parameter to indicate whether the explainer can support multiple outputs in one running, default is False.
    
    Output:
        sal_dict - CWOX results: A dictionary of cluter heatmaps and class heatmaps 
        (keys: 'clusterx', ... , 'clusterx_y',... where x is the cluster id and y is the class id).

    Example

    >>> import torch
    >>> from torchvision import datasets, models, transforms, utils
    >>> from PIL import Image
    >>> from CWOX.CWOX import CWOX
    >>> from CWOX.IOX import IOX
    >>> from CWOX.plt_wox import plot_cwox

    >>> loader = transforms.Compose([
    >>>         transforms.Resize((224,224)),
    >>>         transforms.ToTensor(),
    >>>         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
    >>> img_path='eval_image//cello_guitar.jpg'
    >>> image = Image.open(img_path).convert('RGB')
    >>> image = loader(image).float()
    >>> image=torch.unsqueeze(image, 0)
    >>> model = models.resnet50(pretrained=True)
    >>> _ = model.train(False) >>> put model in evaluation mode

    >>> from CWOX.grad_cam_cwox import grad_cam_cwox
    >>> IOX_cluster=IOX(grad_cam_cwox(model,layer='layer4'))
    >>> IOX_class=IOX(grad_cam_cwox(model,layer='layer3.5.relu'))
    >>> cluster_use_final=[[486,889],[402,420,546]]
    >>> sal_dict=CWOX_2s(image,cluster_use_final,cluster_method=IOX_cluster,class_method=IOX_class,delta=50,multiple_output=False)
    """
    
    sal_dict={}
    top_k=[]
    for label in cluster_list:
        top_k.extend(label)


    ########################################################
    # Get the cluster_explain_list and class_explain_list
    cluster_explain_list=[]
    class_explain_list=[]

    '''
    cluster_list: 3 situations
    [[1,2],[3,4]]
    [[1,2,3,4]]
    [[1],[2]]
    '''

    for i in cluster_list:
        if len(i)==len(top_k):
            cluster_explain_list.append(i)
            for j in i: class_explain_list.append(j)

        if len(i)==1:
            cluster_explain_list.append(i)

        if len(i)>1 and len(i)<len(top_k):
            cluster_explain_list.append(i)
            for j in i: class_explain_list.append(j)

    ########################################################
    # Produce the cluster heatmaps and class heatmaps respectively
    cluster_dict={}
    class_dict={}

    if multiple_output==False:
        for cluster in cluster_explain_list:
            cluster_result=cluster_method(image,cluster)
            if type(cluster_result) is np.ndarray==False:
                cluster_result=cluster_result.asnumpy()
            if len(cluster_result.shape)==4:
                cluster_result=cluster_result[0,0]
            if len(cluster_result.shape)==3:
                cluster_result=cluster_result[0]
            cluster_dict[str(cluster)]=cluster_result
            

        for class1 in class_explain_list:
            class_result=class_method(image,class1)
            if type(class_result) is np.ndarray==False:
                class_result=class_result.asnumpy()
            if len(class_result.shape)==4:
                class_result=class_result[0,0]
            if len(class_result.shape)==3:
                class_result=class_result[0]
            class_dict[str(class1)]=class_result

    if multiple_output==True:
        cluster_result_list=cluster_method(image,cluster_explain_list)

        for cluster_ind in range(len(cluster_explain_list)):
            cluster_result=cluster_result_list[cluster_ind]
            if type(cluster_result) is np.ndarray==False:
                cluster_result=cluster_result.asnumpy()
            cluster_dict[str(cluster_explain_list[cluster_ind])]=cluster_result

        class_result_list=class_method(image,class_explain_list)

        for class_ind in range(len(class_explain_list)):
            class_result=class_result_list[class_ind]
            if type(class_result) is np.ndarray==False:
                class_result=class_result.asnumpy()
            class_dict[str(class_explain_list[class_ind])]=class_result
        

    ########################################################
    # From the cluster heatmaps and class heatmaps to build the CWOX result.
    for i in range(len(cluster_list)):
        class_use=cluster_list[i]
        class_not_use=[]
        for j in [x for x in range(len(cluster_list)) if x != i]:
            class_not_use.append(cluster_list[j])
    
        saliency1=one_v_one(class_use,class_not_use,cluster_dict)
        saliency1=np.clip(saliency1,a_min=0,a_max=10000)
        name1='cluster'+str(i+1)
        sal_dict[name1]=saliency1


        if len(cluster_list)==1:
            for k in range(len(class_use)):
                poss=class_use[k]
                class_use=np.asarray(class_use)
                mask = np.ones(len(class_use), bool)
                mask[k] = False
                negs= list(class_use[mask])
                class_use=list(class_use)
                saliency2=one_v_one(poss,negs,class_dict)
                temp11=np.percentile(saliency1,delta)
                saliency_support =np.where(saliency1 >temp11, 1, 0)
                saliency3=saliency_support *saliency2
                name2='cluster'+str(i+1)+'_'+str(class_use[k])
                sal_dict[name2]=np.clip(saliency3,a_min=0,a_max=10000)

        if len(cluster_list)>1 and len(class_use)>1:
            for k in range(len(class_use)):
                poss=class_use[k]
                class_use=np.asarray(class_use)
                mask = np.ones(len(class_use), bool)
                mask[k] = False
                negs= list(class_use[mask])
                class_use=list(class_use)
                saliency2=one_v_one(poss,negs,class_dict)
                temp11=np.percentile(saliency1,delta)
                saliency_support =np.where(saliency1 >temp11, 1, 0)
                saliency3=saliency_support *saliency2
                name2='cluster'+str(i+1)+'_'+str(class_use[k])
                sal_dict[name2]=np.clip(saliency3,a_min=0,a_max=10000)        
    return sal_dict
