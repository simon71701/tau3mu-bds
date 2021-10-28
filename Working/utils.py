import numpy as np
import pandas as pd
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float, gamma: float, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
        
def get_idx_for_interested_fpr(fpr, interested_fpr):
    res = []
    for i in range(1, fpr.shape[0]):
        if fpr[i] > interested_fpr:
            res.append(i-1)
            break
    return res
    
def augment(dataframe):
    copy_origin = dataframe.copy()
    addition = dataframe.copy()
    
    if 'gen_mu_phi' in dataframe.keys():
        copy_origin['gen_mu_phi'] = copy_origin['gen_mu_phi'] % (2*math.pi)
        addition['gen_mu_phi'] = (addition['gen_mu_phi']+math.pi) % (2*math.pi)
        
    copy_origin['mu_hit_sim_phi'] = copy_origin['mu_hit_sim_phi'].map(lambda x: np.radians(x%360))
    addition['mu_hit_sim_phi'] = addition['mu_hit_sim_phi'].map(lambda x: np.radians((x%360)+180))
    
    
    new_frame = pd.concat((copy_origin, addition))
    
    return new_frame
    
def filterandpad(dataframe, maxhits, variables, one_endcap=False):
    
    dataset = []
    for i in tqdm(range(len(dataframe)), desc='Filtering Dataset'):
        
        event = dataframe.iloc[i]
        #print(event['n_gen_tau'])
        
        if 'n_gen_tau' in event.keys():
            if int(event['n_gen_tau']) > 1:
                continue
        

        if int(event['n_mu_hit']) == 0:
            continue
        
        new_event = []
        
        unfiltered_event = [event[key] for key in variables]
        stations = event['mu_hit_station']
        is_neighbor = event['mu_hit_neighbor']
        
        for char in unfiltered_event:
            new_char = []
            for idx, hit in enumerate(char):
                if stations[idx] == 1 and is_neighbor[idx] == 0:
                    new_char.append(hit)
                
            while len(new_char) < maxhits:
                new_char.append(0)
                
            if len(new_char) > maxhits:
                new_char = new_char[0:maxhits]
            
            new_event.append(new_char)
        
        new_event = np.array(new_event)
        new_event = new_event.flatten()
        dataset.append(new_event)
        
    return dataset    

def valid_filterandpad(dataframe, maxhits, variables, one_endcap=False):
    
    dataset = []
    for i in tqdm(range(len(dataframe)), desc='Filtering Dataset'):
        
        event = dataframe.iloc[i]

        if int(event['n_mu_hit']) == 0:
            continue
        
        new_event = []
        
        unfiltered_event = [event[key] for key in variables]
        stations = event['mu_hit_station']
        is_neighbor = event['mu_hit_neighbor']
        
        for char in unfiltered_event:
            new_char = []
            for idx, hit in enumerate(char):
                if stations[idx] == 1 and is_neighbor[idx] == 0:
                    new_char.append(hit)
                
            while len(new_char) < maxhits:
                new_char.append(0)
                
            if len(new_char) > maxhits:
                new_char = new_char[0:maxhits]
            
            new_event.append(new_char)
        
        new_event = np.array(new_event)
        new_event = new_event.flatten()
        dataset.append(new_event)
        
    return dataset

    
    
    
    
    
    
    
    
    
    
    
