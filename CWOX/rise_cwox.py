import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.autograd import Function, Variable
from torchvision import datasets, models, transforms, utils
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
import os, sys, copy ; sys.path.append('..')
from PIL import Image
from skimage.transform import resize

# Function to generate masks at one time
def generate_masks( N, s, p1,input_size):
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size
    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')
    masks = np.empty((N, *input_size))
    for i in range(N):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                     anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    masks = masks.reshape(-1, 1, *input_size)
    masks = torch.from_numpy(masks).float().to(device="cuda")
    return masks

class rise_cwox():
    r"""
    Produces cluster heatmaps with **RISE** as base explainer. The predicted target is changed to: 

    P(\mathbf{C})=\sum_{c \in \mathbf{C}}P(c)), where c is the class in the cluster **C**. 

    Args:
        model (torch.nn) – The black-box model to be explained.
        N (Int) - The number of masks to be sampled.
        mask_probability (Float 0-1) - The ratio of inputs to be masked.
        down_sample_size (Int) - The original size of binary masks.
        gpu_batch (Int) - The number of masked images processed to make predctions in each batch, default is 32.
        activation_fn (torch.nn) - The activation layer that is applied to transform logits into probabilities, default is nn.Softmax(dim=1).
    Inputs:
        image (Tensor) - The input image to be explained, a 2D tensor of shape (H,W).
        targets (List of List of Int) - The clusters to be explained. Since RISE itself supports the multiple outputs for different target classes, 
        the converted cluster version can also supports that. e.g. [[0,1,2], [3,4,5]] means the cluster that includes class 0,1,2 and cluster that includes class 3,4,5 are explained.
    Output:
        a 3D array of shape (len(targets),H,W) that represents the saliency value in each pixel position for different clusters.
    
    """
    def __init__(self, model,N,mask_probability,down_sample_size,gpu_batch=32,activation_fn=nn.Softmax(dim=1)):
        self.model=model
        self.N=N
        self.mask_probability=mask_probability
        self.down_sample_size=down_sample_size
        self.gpu_batch=gpu_batch
        self.model = nn.Sequential(model, activation_fn)
        self.model = self.model.eval()
        self.model = self.model.cuda()
        for p in self.model.parameters():
            p.requires_grad = False

    def __call__(self, image, targets):
        image = image.cuda()
         #Generate Masks first
        _, _, H, W = image.shape
        input_size=(H,W)
        masks=generate_masks(self.N, self.down_sample_size, self.mask_probability, input_size)
        # Apply array of filters to the image
        stack = torch.mul(masks, image)
        p = []   
        for i in range(0, self.N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, self.N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), masks.view(self.N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / self.N / self.mask_probability
        sal=sal.cpu().numpy()
        sal_output=[]
        for class1 in targets:
            temp_sal=sal[class1]
            temp_sal=temp_sal
            if len(class1)==1:
                temp_sal=(temp_sal-temp_sal.min())/(temp_sal.max()-temp_sal.min())
                sal_output.append(temp_sal[0])
            if len(class1)>1:
                temp_sal=np.sum(temp_sal,axis=0)
                temp_sal=(temp_sal-temp_sal.min())/(temp_sal.max()-temp_sal.min())
                sal_output.append(temp_sal) 
        return sal_output
