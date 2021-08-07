import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.autograd import Function, Variable
from torchvision import datasets, models, transforms, utils
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
from scipy.special import softmax
import os, sys, copy ; sys.path.append('..')
from .base_explainer.attribution.gradient import gradient
from .base_explainer.attribution.excitation_backprop import excitation_backprop


class excitationbackprop_cwox():
    r"""

    Produces cluster heatmaps with **ExcitationBackprop** as base explainer. The back-propagation target is changed to: 
    z_{\textbf{C}}=log\sum_{c\in \textbf{C}}{e}^{z_c}, where c is the class in the cluster **C**

    Args:
    model (torch.nn) – The black-box model to be explained.
    layer (Layer Name) - The pivoting layer used.

    Inputs:
    image (Tensor) - The input image to be explained, a 2D tensor of shape (H,W).
    targets (List of Int) - The cluster to be explained, e.g. [0,1,2] means the cluster that includes class 0,1,2 is explained.

    Output:
    sal (Array) - a 2D array of shape (H,W) that represents the saliency value in each pixel position.
    """

    def __init__(self, model, layer=""):
        self.model=model
        self.layer=layer

    def __call__(self, image, targets):
        """Call function for `excitationbackprop_cwox`."""
        poss=targets
        probs = self.model(image).softmax(-1)
        grad = torch.zeros_like(probs)
        poss_p = probs[0, poss]
        grad[0, poss] = poss_p / poss_p.sum()
        saliency = excitation_backprop(self.model, image, grad, saliency_layer=self.layer,resize = True)[0,0]
        saliency=saliency.detach().cpu().numpy()
        return saliency
