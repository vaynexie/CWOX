import os
import sys
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

class_labels = []
with open(os.path.join(__location__, 'labels.txt'), 'r') as f:
    for line in f:
        class_labels.append(line[:-1].split(',')[0])

def imsc(img):
    lim = [img.min(), img.max()]
    img = img - lim[0] 
    img*=(1 / (lim[1] - lim[0]))
    img = np.clip(img, a_min=0, a_max=1)
    bitmap = np.transpose(img,(1, 2, 0)) 
    return bitmap


def plot_cwox(result_dict,image,cluster_list):
    r"""
    Function of making the plot for the CWOX results based on the output saliency dictionary.

    Inputs:
        result_dict (Dictionary) - resulted dictionary from CWOX explanation.
        image (Tensor) - The input image to be explained, a 2D tensor of shape (H,W).
        cluster_list (List of List) - The cluster information that is used to make explanation. 
        e.g. [[0,1],[2,3]]: in this case class 0 and 1 are in the same cluster, and class 2 and 3 are in another cluster.
    """
    img=imsc(image[0])
    sal_dict_display=result_dict
    k_t=[]
    for i in cluster_list:
        k_t.extend(i)
    rcParams['figure.figsize'] = 6,3
    figsize=(8, 5)
    subplot_size0=3.5
    subplot_size1=3.5

    if len(cluster_list)==k_t:
        fig, ax = plt.subplots(1, k_t, figsize=(subplot_size0*k_t,subplot_size1*k_t))
        for i in range(len(cluster_list)):
            heatmap=sal_dict_display['cluster'+str(i+1)]
            ax[i].imshow(img)
            ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
            ax[i].axes.get_xaxis().set_visible(False)
            ax[i].axes.get_yaxis().set_visible(False)
            ax[i].set_title('Cluster'+str(i+1)+'-'+str(class_labels[cluster_list[i][0]]))


    if len(cluster_list)!=k_t:
        for i in range(len(cluster_list)):

            class_use=cluster_list[i]
            class_not_use=[]

            for j in [x for x in range(len(cluster_list)) if x != i]:
                    class_not_use+=cluster_list[j]

            if len(class_use)>1:
                fig, ax = plt.subplots(1, len(class_use)+1, figsize=(subplot_size0*(len(class_use)+1),subplot_size1*(len(class_use)+1)))
                heatmap=sal_dict_display['cluster'+str(i+1)]
                ax[0].imshow(img)
                ax[0].imshow(heatmap, cmap='jet', alpha=0.5)
                ax[0].axes.get_xaxis().set_visible(False)
                ax[0].axes.get_yaxis().set_visible(False)
                ax[0].set_title('Cluster'+str(i+1))
                
                for k in range(len(class_use)):
                    heatmap=sal_dict_display['cluster'+str(i+1)+'_'+str(class_use[k])]
                    ax[k+1].imshow(img)
                    ax[k+1].imshow(heatmap, cmap='jet', alpha=0.6)
                    ax[k+1].axes.get_xaxis().set_visible(False)
                    ax[k+1].axes.get_yaxis().set_visible(False)
                    ax[k+1].set_title(str(class_labels[class_use[k]]))

            if len(class_use)==1:
                fig, ax = plt.subplots(1, 1, figsize=(subplot_size0*len(class_use),subplot_size1*len(class_use)))
                heatmap=sal_dict_display['cluster'+str(i+1)]         
                ax.imshow(img)
                ax.imshow(heatmap, cmap='jet', alpha=0.6)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_title('Cluster'+str(i+1)+'-'+str(class_labels[class_use[0]]))

# def plot_swox(result_list,image,top_k):
#     rcParams['figure.figsize'] = 6,3
#     figsize=(8, 5)
#     subplot_size0=3.5
#     subplot_size1=3.5
#     k_t=len(top_k)
#     fig, ax = plt.subplots(1, k_t, figsize=(subplot_size0*k_t,subplot_size1*k_t))
#     img=imsc(image[0])
#     for i in range(len(top_k)):
#         category_id=top_k[i]
#         saliency=result_list[i]
#         saliency=np.clip(saliency,a_min=0,a_max=None)
#         ax[i].imshow(img)
#         ax[i].imshow(saliency, cmap='jet', alpha=0.6)
#         ax[i].axes.get_xaxis().set_visible(False)
#         ax[i].axes.get_yaxis().set_visible(False)
#         ax[i].set_title(str(class_labels[category_id]))
