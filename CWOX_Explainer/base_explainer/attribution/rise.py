import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from torch.autograd import Function, Variable
from torchvision import datasets, models, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os, sys, copy ; sys.path.append('..')
from PIL import Image
from scipy.special import softmax
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.nn.functional import conv2d
cudnn.benchmark = True
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize



def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))



class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')
        self.masks = np.empty((N, *self.input_size))
        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1
    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]
    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        masks1=self.masks.view(N, H * W)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        p1_value=self.p1
        return sal,p,masks1,N,p1_value,CL




def norm_data(list1):
    max_value=np.max(list1)
    min_value=np.min(list1)
    for i in range(len(list1)):
      for j in range(len(list1[i])):
        list1[i][j]=(list1[i][j]-min_value)/(max_value-min_value)
    return list1


def cwox_step(pos_id,p,N,p1,CL,mask):
  if isinstance(pos_id, np.int64)==True or type(pos_id)==int or isinstance(pos_id, np.int32)==True:pos_id=[pos_id]
  p_np=list(p.cpu().numpy())
  for i in range(len(p_np)):
    temp_pos=0   
    for pos in pos_id:
      temp_pos+=p_np[i][pos]
    p_np[i]=np.append(p_np[i],temp_pos)
  p_modify=torch.FloatTensor(p_np)
  p_modify=p_modify.to(device="cuda")
  mask1=mask.to(device="cuda")
  sal = torch.matmul(p_modify.data.transpose(0, 1), mask1.view(N, 224 *224))
  sal = sal.view((1001, 224, 224))
  sal = sal / N / p1
  sal=sal.cpu().numpy()
  sal1=sal[1000]
  sal1=norm_data(sal1)
  return sal1