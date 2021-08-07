from mindspore.explainer.explanation import CWOX, grad_cam_cwox, excitationbackprop_cwox, RISE_quick, rise_cwox, LIME, lime_cwox

#GradCAM

def grad_cam(x, model, clusters, cluster_layer, class_layer):
    def cluster_method(inputs,targets):
        grad_cam_cluster=grad_cam_cwox(model, cluster_layer)
        return grad_cam_cluster(inputs,targets)

    def class_method(inputs,targets):
        grad_cam_class=grad_cam_cwox(model, class_layer)
        return grad_cam_class(inputs,targets)
    sal_dict=CWOX(x, clusters, cluster_method=cluster_method, class_method=class_method, delta=0, multiple_output=False)
    return sal_dict

def mwp(x, model, clusters, cluster_layer, class_layer):
    def cluster_method(inputs,targets):
        excitationbackprop_cluster=excitationbackprop_cwox(model,layer=cluster_layer)
        return excitationbackprop_cluster(inputs,targets)

    def class_method(inputs,targets):
        excitationbackprop_class=excitationbackprop_cwox(model,layer=class_layer)
        return excitationbackprop_class(inputs,targets)
    sal_dict=CWOX(x, clusters, cluster_method=cluster_method, class_method=class_method, delta=0, multiple_output=False)
    return sal_dict

def rise(x, model, clusters, cluster_size= 15, cluster_prob= 0.3, class_size= 10, class_prob= 0.14):
    def cluster_method(inputs,targets):
        rise_cluster=rise_cwox(model,N=5000,mask_probability=cluster_prob, down_sample_size=cluster_size, gpu_batch= 20)
        return rise_cluster(inputs,targets)

    def class_method(inputs,targets):
        rise_class=RISE_quick(model,N=3000,mask_probability=class_prob, down_sample_size=class_size, gpu_batch=20)
        return rise_class(inputs,targets)

    sal_dict=CWOX(x, clusters, cluster_method=cluster_method, class_method=class_method, delta=0, multiple_output=True)
    return sal_dict

from utils import to_PilTempPath
import os

def lime(x, model, clusters, kernel_size = 5):
    def cluster_method(inputs,targets):
        lime_cluster=lime_cwox(model,kernel_size= kernel_size,number_sample=2000,gpu_batch=100)
        return lime_cluster(inputs,targets)

    def class_method(inputs,targets):
        lime_class=LIME(model, kernel_size=kernel_size,number_sample=2000,gpu_batch=100)
        return lime_class(inputs,targets)

    path = to_PilTempPath(x[0])
    sal_dict=CWOX(path, clusters, cluster_method=cluster_method, class_method=class_method, delta=0, multiple_output=True)
    os.remove(path)
    return sal_dict
