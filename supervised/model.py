"""
Model architecture

"""

import pandas as pd
import numpy as np

from torch import Tensor
from torch import nn
import torch

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

import glob
import torch
import torch.nn as nn
from random import sample
from torch.nn.modules.linear import Linear
import torchvision.models as models
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

## Convolutional Layer Definition
class ConvEncoder(nn.Module):
    """
    Convolutional architecutre used for model training
    
    Conv-Relu-Conv-Conv-MaxPool-FC
    
    """

    def __init__(self, num_feats, num_hiddens=8, p_drop=0.0,
                kernel_size=1000, conv_stride=5, 
                pool_stride=5, dilation=2, embed_size=4311):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, stride=conv_stride, 
                      kernel_size=kernel_size, dilation=dilation, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=1, out_channels=1, stride=conv_stride,
                      kernel_size=kernel_size, dilation=dilation, bias=False),
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=1, stride=conv_stride,
                      kernel_size=kernel_size, dilation=dilation, bias=False),
            nn.BatchNorm1d(1),
            nn.MaxPool1d(kernel_size=kernel_size, stride=pool_stride, padding=0, 
                         dilation=1)
        )
        
        self.fc1 = nn.Linear(embed_size, num_hiddens)
        
    def forward(self, data: Tensor):
        
        embed = self.encoder(data.unsqueeze(1))
        #breakpoint()
        x = self.fc1(embed)
        return x

## Overall Model: EpiShift 
class EpiShift(nn.Module):
    """
    Training Alzheimer's prediction model, with the option of using 
    a pre-trained convolutional layer to enable transfer learning
    """
        
    def __init__(self, conv_stride=2, kernel_size=500, dilation=2, 
                 embed_size=4302, pretrain_encoder=None):
        super(EpiShift, self).__init__()
        
        self.encoder = pretrain_encoder.encoder
        self.fc1 = nn.Linear(embed_size, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, data: Tensor):
        
        encoded = self.encoder(data)
        x = self.fc1(encoded)
        x = self.fc2(x)
        return x