"""
Dataloders

"""

import pandas as pd
import numpy as np

from torch import Tensor
from torch import nn
import torch

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

from sklearn.model_selection import train_test_split
import glob
import torch
import torch.nn as nn
from random import sample
from torch.nn.modules.linear import Linear
import torchvision.models as models
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Rush_DNAse_Dataset(torch.utils.data.Dataset):
    """
    Set up dataloaders
    """
    
    def __init__(self, fnames, label_file):
        'Initialization'
        
        tracks = []
        file_ids = []
        
        for f in fnames:
            read_in = torch.load(f)
            if len(read_in)==2875012:
                tracks.append(read_in)
                file_ids.append(f.split('/')[-1].split('.')[0])
            
        # This attribute contains the genomic data
        self.tracks = torch.stack(tracks)
        
        # This attribute contains label information
        self.labels = label_file.set_index('file_accession').loc[file_ids,'label'].values
        self.labels = Tensor((self.labels=='AD').astype('float'))
            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.tracks[index], self.labels[index]
    
def get_train_val_test(fold_num, labels): 
    """
    Read in fold information from Sebastian's code
    
    """
    
    fold = pd.read_csv('torch_data/rush/new/DNase_fold_'+
                       str(fold_num)+'_labels.csv', index_col=0)
    
    test_ids = fold.index.values
    train_ids = list(set(labels['file_accession'].values).difference(set(test_ids)))
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, 
                                  random_state=10)
    
    return train_ids, val_ids, test_ids

def get_fnames(fnames, ids):
    """
    Get corresponding torch Tensor format filenames for each sample
    """
    
    filename2id_map = {f.split('/')[-1].split('.')[0]:idx  
                     for idx, f in enumerate(fnames)}
    
    filename_idx = []
    for f in ids:
        try:
            filename_idx.append(filename2id_map[f])
        except:
            print('Not found: ',f)
    return np.array(fnames)[filename_idx]