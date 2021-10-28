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

def genNpArrays():
    PU200_file = open("DsTau3muPU200_MTD.pkl", "rb")
    SignalPU200 = pickle.load(PU200_file)
    PU200_file.close()

    MinBias_file = open("MinBiasPU200_MTD.pkl", "rb")
    BgPU200 = pickle.load(MinBias_file)
    MinBias_file.close()
    
    SignalPU200['gen_mu_phi'] = SignalPU200['gen_mu_phi'] % (2*math.pi)
    SignalPU200['mu_hit_sim_phi'] = SignalPU200['mu_hit_sim_phi'].map(lambda x: np.radians(x%360))
    SignalPU200['mu_hit_sim_theta'] = SignalPU200['mu_hit_sim_theta'].map(lambda x: np.radians(x%360))
    
    BgPU200['mu_hit_sim_phi'] = BgPU200['mu_hit_sim_phi'].map(lambda x: np.radians(x%360))
    BgPU200['mu_hit_sim_theta'] = BgPU200['mu_hit_sim_theta'].map(lambda x: np.radians(x%360))
    
    train_signal = SignalPU200[0:int(.95*len(SignalPU200))]
    valid_signal= SignalPU200[96519:len(SignalPU200)]
    
    interested_vars = ['mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_theta', 'mu_hit_sim_z', 'mu_hit_bend', 'mu_hit_ring', 'mu_hit_quality']
    
    train_signal_npy = filterandpad(train_signal, 50, interested_vars)
    valid_signal_npy = valid_filterandpad(valid_signal, 50, interested_vars)
    
    np.save('SignalPU200_Train.npy', train_signal_npy)
    np.save('SignalPU200_Valid.npy', valid_signal_npy)
    
    print("Signal Completed")
    
    train_bg = BgPU200[0:int(4.92*len(train_signal_npy))]
    valid_bg = BgPU200[int(4.92*len(train_signal_npy)):]
    
    bg_train_npy = filterandpad(train_bg, 50, interested_vars)
    bg_valid_npy = valid_filterandpad(valid_bg, 50, interested_vars)
    
    np.save('/scratch/gilbreth/simon73/Working/BgPU200_Train.npy', bg_train_npy)
    np.save('/scratch/gilbreth/simon73/Working/BgPU200_Valid.npy', bg_valid_npy)
    
    print("Background Completed. Done")
    
    
    
    
    
    
    
    
    
    
    
