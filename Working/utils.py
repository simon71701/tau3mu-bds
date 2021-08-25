import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import argparse
import numpy as np

def arg_parse():
    parser = argparse.ArgumentParser(description='FCN Arguments.')

    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float,
                       help='Learning Rate')
    parser.add_argument('--pileup', type=int,
                       help='Pileup amount. Options: 0, 200')
    parser.add_argument('--num_funnel_layers', type=int,
                       help='Number of Funneling Layers')
    parser.add_argument('--maxhits', type=int,
                        help="Maximum number of hits allowed")
    parser.add_argument('--extra_filter', type=int,
                        help="Filter for pt and eta. Options: 1 for true, 0 for false")
    parser.add_argument('--mix', type=float,
                        help='If >0, denotes the fraction with which the samples are mixed. See documentation of the function mixData for more details.')
    parser.add_argument('--test_on', type=int,
                        help="Denotes which dataset the model should be tested on. Options: 0, 200")
    parser.add_argument('--early_stop', type=int,
                        help="Denotes whether training will use early stopping. 1 for true, 0 for false")


    parser.set_defaults(batch_size=50,
                        dropout=0.0,
                        epochs=250,
                        lr=.0001,
                        pileup=0,
                        num_funnel_layers=5,
                        maxhits=72,
                        extra_filter=0,
                        mix=0,
                        test_on=200,
                        early_stop=0)

    return parser.parse_args()

# Defines the loss function used by the model
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

# Function that takes in the name of a variable and outputs a dictionary linking that variable to its corresponding index in the dataset
# Ex: dictionary = make_lookup(taugunvars)
# print(dictionary['n_mu_hit']) would output 2
def make_lookup(variables):
    lookup = dict({})
    for i in range(len(variables)):
        lookup.update({variables[i]:i})
        
    return lookup

## Filter for events that only have 1 tau
def filterPU200(dataset):
    singledOut = []
    
    for event in dataset:
        if event[0] == 1:
            singledOut.append(event)
    
    return singledOut

# Filter for events that have up to a specified number of muons and pad events with zeros. Characteristics are ordered in the same way as interested_vars
def filterandpad(dataset, maxhits, variables, interested_vars, pt_eta_filter, one_endcap=False):
    filtered = []
    lookup = make_lookup(variables)
    
    var_indices = [lookup[var] for var in interested_vars]
    station_index = lookup['mu_hit_station']
    neighbor_index = lookup['mu_hit_neighbor']
    
    event_indices = []
       
    # For each event in the dataset, construct a new event of the same structure that only has the specified characteristics in interested_vars.
    # Then, for each of these "new events", we pad each characteristic with 0's so that each event is the exact same shape with length len(interested_vars)*maxhits
    for event in dataset:
        chars = []
        new_event = [[] for i in range(len(var_indices))]
        for i in range(len(event[station_index])):
            if one_endcap==False:
                if event[station_index][i] == 1 and event[neighbor_index][i] == 0: # If muon hit was in the first station
                    for j in range(len(var_indices)):
                        new_event[j].append(event[var_indices[j]][i])
            
            elif one_endcap==True:
                if (event[station_index][i] == 1) and (event[neighbor_index][i] == 0) and (event[lookup['mu_hit_sim_eta']][i] >= 0): # If muon hit was in the first station and pos endcap
                    for j in range(len(var_indices)):
                        new_event[j].append(event[var_indices[j]][i])

        if len(new_event[0]) == 0:
            continue
        
        if len(new_event[0]) <= maxhits:
            for i in range(len(new_event)):
                char = np.pad(new_event[i], (0,maxhits-len(new_event[i])), 'constant', constant_values=float(0))
                for muon in char:
                    chars.append(float(muon))
            
            chars = np.array(chars)
            filtered.append(chars)
            event_indices.append(count)
        
    filtered = np.array(filtered)
    return filtered, event_indices

# See filterandpad
def minbias_filterandpad(minbias, maxhits, minbiasvars, interested_vars, one_endcap=False):
    filtered = []
    
    lookup = make_lookup(minbiasvars)
    
    var_indices = [lookup[var] for var in interested_vars]
    hits_index = lookup['n_mu_hit']
    station_index = lookup['mu_hit_station']
    neighbor_index = lookup['mu_hit_neighbor']
    
    lens = []
    for event in minbias:
        chars = []
        new_event = [[] for i in range(len(var_indices))]
        for i in range(len(event[station_index])):
            if one_endcap==False:
                if event[station_index][i] == 1 and event[neighbor_index][i] == 0: # If muon hit was in the first station
                    for j in range(len(var_indices)):
                        new_event[j].append(event[var_indices[j]][i])
            
            elif one_endcap==True:
                if (event[station_index][i] == 1) and (event[neighbor_index][i] == 0) and (event[lookup['mu_hit_sim_eta']][i] <= 0): # If muon hit was in the first station
                    for j in range(len(var_indices)):
                        new_event[j].append(event[var_indices[j]][i])

        if len(new_event[0]) == 0:
            continue
        
        if len(new_event[0]) <= maxhits:
            for i in range(len(new_event)):
                char = np.pad(new_event[i], (0,maxhits-len(new_event[i])), 'constant', constant_values=float(0))
                for muon in char:
                    chars.append(float(muon))
            
            chars = np.array(chars)
            filtered.append(chars)

    return filtered 

# By Siqi Miao
def get_idx_for_interested_fpr(fpr, interested_fpr):
    res = []
    for i in range(1, fpr.shape[0]):
        if fpr[i] > interested_fpr:
            res.append(i-1)
            break
    return res
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
