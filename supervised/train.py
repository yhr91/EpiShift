"""
Training functions

"""

import pandas as pd
import numpy as np

from torch import Tensor
from torch import nn
import torch
import argparse

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

from sklearn.model_selection import train_test_split, KFold

import glob
import torch
import torch.nn as nn
from random import sample
from torch.nn.modules.linear import Linear
import torchvision.models as models
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from data import Rush_DNAse_Dataset, get_train_val_test, get_fnames
from model import ConvEncoder, EpiShift



def train_epishift(model, train_dataloader, val_dataloader, 
                   num_epochs=100, device=None,
                  pretrain_encoder=None, lr=1e-3):
    """
    Training EpiShift model
    """
    
  
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, 
                           weight_decay=5e-4)
    #scheduler = StepLR(optimizer, step_size=1, 
    #                   gamma=0.5)
    min_val_loss = np.inf
    m = nn.Sigmoid()
    best_model = deepcopy(model)
    model.train()
    model.to(device)
    losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(epoch)
        model.train()

        # Training step
        for step, data in enumerate(train_dataloader):
            X = data[0].to(device)
            y = data[1].to(device)
            optimizer.zero_grad()

            pred = model(X.unsqueeze(1))
            loss = criterion(m(pred).squeeze(), y)
            loss.backward()
            optimizer.step()

            losses.append(loss)
            #print("Training Loss:", loss.detach().cpu().numpy())

        # Validation step
        val_loss_sum = 0
        model.eval()
        for step, data in enumerate(val_dataloader):
            X = data[0].to(device)
            y = data[1].to(device)

            val_pred = model(X.unsqueeze(1))
            val_loss = criterion(m(val_pred).squeeze(), y)
            val_loss_sum += val_loss

        # Check performance against validation set
        print("Validation Loss:", val_loss_sum.detach().cpu().numpy())
        if val_loss_sum < min_val_loss:
            min_val_loss = val_loss_sum
            best_model = deepcopy(model)
            
    return best_model
            

def test_epishift(model, test_dataloader, device=None):
    """
     Testing model performance
    """
    test_preds = []
    ys = []
    model.eval()
    m = nn.Sigmoid()
    model.to(device)
    
    # Testing step
    for step, data in enumerate(test_dataloader):
        X = data[0].to(device)
        y = data[1].to(device)

        test_pred = m(model(X.unsqueeze(1))).squeeze()
        tp = test_pred.squeeze().cpu().detach().numpy()
        test_preds.extend(tp)
        ys.extend(y.cpu().detach().numpy())
        
    return {'pred':test_preds, 
            'truth':ys}

  
def trainer(args):
    """
    Setting up dataloader, train/test split,
    k fold cross validation, model saving
    
    """
    x_dim = 2881021
    model_dim = 8
    batch_size = 10
    device = args['device']
    modality = args['modality']
    pretrain= args['pretrain']

    labels = pd.read_csv('./experiment_labels.csv')
    if modality == 'DNase-seq':
        labels_subset = labels[labels['modality']==modality]
    elif modality is not None:
        labels_subset = labels[labels['target']==modality]
    else:
        labels_subset = labels
    rush_fnames = glob.glob('./torch_data/rush/new/*.data')
    patient_ids = labels_subset['subject_id'].unique()

    batch_size=50
    num_epochs = args['num_epochs']
    lr= args['lr']

    kf = KFold(n_splits=5, random_state=10, shuffle=True)
    results = {}
    
    idx = 0
    for train_patient, test_patient in kf.split(patient_ids):
        print('Split '+ str(idx+1))

        # Use pre-trained weights from all 
        #weights = torch.load('nopre_allmodal_5split_'+str(idx))
        pretrain_encoder = ConvEncoder(num_feats=x_dim)
        model = EpiShift(pretrain_encoder=pretrain_encoder)
        #model.load_state_dict(weights)

        # The other option to try is to freeze the first two layers

        train_patients = patient_ids[train_patient]
        test_patients = patient_ids[test_patient]

        train_ids = labels_subset[labels_subset['subject_id'].isin(train_patients)]['file_accession'].values
        test_ids = labels_subset[labels_subset['subject_id'].isin(test_patients)]['file_accession'].values
        train_ids, val_ids = train_test_split(train_ids, test_size=0.1, 
                                      random_state=10)

        train_ds = Rush_DNAse_Dataset(get_fnames(rush_fnames, train_ids),
                                      label_file=labels_subset)
        val_ds = Rush_DNAse_Dataset(get_fnames(rush_fnames, val_ids),
                                      label_file=labels_subset)
        test_ds = Rush_DNAse_Dataset(get_fnames(rush_fnames, test_ids),
                                      label_file=labels_subset)

        train_loader = torch.utils.data.DataLoader(train_ds,
                                batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds,
                                batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds,
                                batch_size=batch_size, shuffle=True)

        best_model = train_epishift(train_dataloader=train_loader, model=model,
                                    val_dataloader=val_loader, num_epochs=num_epochs,
                                    lr=lr, pretrain_encoder=pretrain_encoder, device=device)
        results[idx] = test_epishift(best_model, test_loader, device=device)
        
        if args['model_save']:
            torch.save(best_model.state_dict(), 'best_model_' + str(modality) + 
                       '_'+ str(pretrain) + '_fold_' +str(idx))
        idx+= 1
        
    np.save('results_' + str(modality) + '_'+ str(pretrain), results)
        
        

def parse_arguments():
    """
    Argument parsing for command line inputs
    
    """

    # dataset arguments
    parser = argparse.ArgumentParser(description='EpiShift')
    parser.add_argument('--modality', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_save', default=True, action='store_false')
    
    return dict(vars(parser.parse_args()))
    
    
if __name__ == "__main__":
    trainer(parse_arguments())
