import torch
import torch.nn as nn
import numpy as np
from base_explainer.attribution.rise import RISE, norm_data

def rise_swox(img, model, labels, size1=10, p1=0.1):
    # Load black box model for explanations
    model = nn.Sequential(model, nn.Softmax(dim=1))
    model = model.eval()
    model = model.cuda()
    for p in model.parameters():
        p.requires_grad = False

    #Generate Mask for cluster
    explainer = RISE(model, (224, 224))
    explainer.generate_masks(N=5000, s=size1, p1=p1)
    with torch.no_grad():
        sal,p,mask,N,p1,CL=explainer(img.cuda())
    return {"label_{}".format(i): sal[i].detach().cpu().numpy() for i in labels}

from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.excitation_backprop import excitation_backprop


def grad_cam_swox(img, model, labels, class_layer):
    return {"label_{}".format(i): grad_cam(
                model, img, i, saliency_layer = class_layer,
                resize = True).detach().cpu().numpy()[0, 0]
            for i in labels
            }

def mwp_swox(img, model, labels, class_layer):
    return {"label_{}".format(i): excitation_backprop(
                model, img, i, saliency_layer = class_layer,
                resize = True).detach().cpu().numpy()[0, 0]
            for i in labels
            }

from lime import lime_image
def inverse_trans(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i, (m, s) in enumerate(zip(mean, std)):
        img[:, i] *= s
        img[:, i] += m
    return img

def trans(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i, (m, s) in enumerate(zip(mean, std)):
        img[:, i] -= m
        img[:, i] /= s
    return img

from lime import lime_image
from torchvision import transforms

def lime_swox(img, model, labels, kernel_size1 = 5, number_sample1 = 2500):
    img = img.clone()
    explainer = lime_image.LimeImageExplainer()
    img = inverse_trans(img).clip(0, 1)
    img_pil = transforms.ToPILImage()(img[0])
    img_np = np.asarray(img_pil)/255
    segmentation_fn = lime_image.SegmentationAlgorithm('quickshift', kernel_size=kernel_size1,
                                        max_dist=10, ratio=1)
    get_tensor = lambda x: trans(torch.tensor(x.transpose((0, 3, 1, 2)), dtype = torch.float32).to(img.device))
    predict = lambda x: model(get_tensor(x)).softmax(-1).detach().cpu().numpy()
    with torch.no_grad():
        explain = explainer.explain_instance(img_np, predict,
                labels = labels, top_labels=None, hide_color =0, num_samples = number_sample1,
            segmentation_fn = segmentation_fn
            )
    sals = {}
    val_tab = np.zeros(explain.segments.max() + 1)
    for i in labels:
        for j, val in explain.local_exp[i]:
            val_tab[j] = val
        sal = val_tab[explain.segments]
        sals["label_{}".format(i)] = sal.clip(0)
    return sals

