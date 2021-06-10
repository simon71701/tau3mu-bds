import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from utils import *
from train import *
from test_model1_4 import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
    
def main():
    
    args = arg_parse()
    
    maxhits = args.maxhits
    interested_vars = ['mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r']
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    
    device = torch.device(dev)
    
    MinBias = open("MinBiasPU200_MTD.pkl", "rb")
    minbias_pu200file = pickle.load(MinBias)
    minbiasvars = minbias_pu200file.keys()
    minbias_pu200 = np.array(minbias_pu200file)
    
    if args.mix == 0:
    
        if args.pileup == 0:
            
            PU0_Private = open("DsTau3muPU0_Private.pkl", "rb")
            tau_pu0file = pickle.load(PU0_Private)
            taugunvars = tau_pu0file.keys()
            taudata = np.array(tau_pu0file)
            
            taudata = np.array(filterandpad(taudata, maxhits, taugunvars, interested_vars, args.extra_filter))
        
            del tau_pu0file
            PU0_Private.close()

        elif args.pileup==200:
            
            PU200_MTD = open("DsTau3muPU200_MTD.pkl", "rb")
            tau_pu200file = pickle.load(PU200_MTD)
            taugunvars = tau_pu200file.keys()
            taudata = np.array(tau_pu200file)
            
            taudata = np.array(filterandpad(taudata, maxhits, taugunvars, interested_vars, args.extra_filter))

            del tau_pu200file
            PU200_MTD.close()
    
    elif args.mix:
        
        PU0_Private = open("DsTau3muPU0_Private.pkl", "rb")
        tau_pu0file = pickle.load(PU0_Private)
        PU200_MTD = open("DsTau3muPU200_MTD.pkl", "rb")
        tau_pu200file = pickle.load(PU200_MTD)
        taugunvars = tau_pu0file.keys()
        
        if args.pileup == 0:
            data1 = np.array(tau_pu0file)
            data2 = np.array(tau_pu200file)
            
        
        elif args.pileup == 200:
            data1 = np.array(tau_pu200file)
            data2 = np.array(tau_pu0file)
        
        data1 = filterandpad(data1, maxhits, taugunvars, interested_vars, args.extra_filter)
        data2 = filterandpad(data2, maxhits, taugunvars, interested_vars, args.extra_filter)
        
        taudata = np.array(mixData((data1, data2), args.mix))
        rng = np.random.default_rng()
        rng.shuffle(taudata)
    
    
    bgdata = np.array(minbias_filterandpad(minbias_pu200,maxhits, minbiasvars, interested_vars))
    
    train_data, test_data = np.split(taudata, [int(.9*len(taudata))])
    
    bgtrain, leftovers = np.split(bgdata, [int(.9*len(taudata))])
    bgtest, leftovers2 = np.split(leftovers, [int(.1*len(taudata))])

    train_labels = torch.ones(np.shape(train_data)[0])
    test_labels = torch.ones(np.shape(test_data)[0])
    
    train_labels = torch.cat((train_labels, torch.zeros(np.shape(bgtrain)[0])))
    test_labels = torch.cat((test_labels, torch.zeros(np.shape(bgtest)[0])))
    
    train_data = np.concatenate((train_data, bgtrain))
    test_data = np.concatenate((test_data, bgtest))

    train_data = torch.tensor(train_data)
    test_data = torch.tensor(test_data)
    
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    del minbias_pu200
    del taudata
    del bgdata
    del train_data
    del test_data

    path = trainModel(train_dataset, test_dataset, maxhits, device, args)
    testModel(args, path)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
