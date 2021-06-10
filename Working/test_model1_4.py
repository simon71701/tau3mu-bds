import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *
from train import *


def testroc(classifier, test_data_loader, path, args):
    classifier.eval()
    
    max_fpr = .001
    
    for batch, labels in test_data_loader:
        with torch.no_grad():
            prediction = torch.flatten(classifier(batch))
            fpr, tpr, thresholds = metrics.roc_curve(labels, prediction)
            roc_auc = metrics.auc(fpr, tpr)
            partial_roc_auc = metrics.roc_auc_score(labels, prediction, max_fpr=max_fpr)
    
    
    index = get_idx_for_interested_fpr(fpr, .01)
    interested_tpr = float(tpr[index])
    
    lw = 2
    plt.clf()
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(max_fpr, interested_tpr, 'ro', label = '.001 Fpr, %.2f Recall (area = %.4f)' % (interested_tpr, partial_roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve on Pileup {0}".format(args.test_on))
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("{0}/TestROC.png".format(path))            
    plt.close()
    

def plotAccuracy(classifier, pu0data, pu200data, bgdata, path, args):
    
    pu0_preds = []
    pu200_preds = []
    bg_preds = []
    
    pu0_loader = torch.utils.data.DataLoader(pu0data, batch_size=1, shuffle=True, drop_last=True)
    pu200_loader = torch.utils.data.DataLoader(pu200data, batch_size=1, shuffle=True, drop_last=True)
    bg_loader = torch.utils.data.DataLoader(bgdata, batch_size=1, shuffle=True, drop_last=True)
    
    for pu0_event in pu0_loader:
        pu0_preds.append(float(classifier(pu0_event)))
    
    for pu200_event in pu200_loader:
        pu200_preds.append(float(classifier(pu200_event)))
    
    for event in bg_loader:
        bg_preds.append(float(classifier(event)))
    
    
        
    plt.hist(pu0_preds, histtype='step', bins= int(.1*len(pu0_preds)**.5), label = "Pileup 0", density=True)
    plt.hist(pu200_preds, histtype='step',bins=int(.1*len(pu200_preds)**.5), label="Pileup 200", density=True)
    plt.hist(bg_preds, histtype='step',bins=int(.05*len(bg_preds)**.5), label="Pileup 200 Background", density=True)
    plt.xticks(ticks = [.1*i for i in range(11)])
    plt.title("Predictions")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend(loc="upper center")
    
    plt.savefig("{0}/AccuracyPlot.png".format(path))
    
    
def testModel(args, path):
    
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
    MinBias.close()
    
    PU0_Private = open("DsTau3muPU0_Private.pkl", "rb")
    tau_pu0file = pickle.load(PU0_Private)
    taugunvars = tau_pu0file.keys()
    pu0data = np.array(tau_pu0file)
        
    PU200_MTD = open("DsTau3muPU200_MTD.pkl", "rb")
    tau_pu200file = pickle.load(PU200_MTD)
    pu200data = np.array(tau_pu200file)
        
    del tau_pu0file
    del tau_pu200file
    PU0_Private.close()
    PU200_MTD.close()
    
    maxhits = args.maxhits
    
    if args.test_on == 0:
        test_data = np.array(filterandpad(pu0data, maxhits, taugunvars, interested_vars, args.extra_filter))
    
    elif args.test_on == 200:
        test_data = np.array(filterandpad(pu200data, maxhits, taugunvars, interested_vars, args.extra_filter))
    
    test_labels = torch.ones(np.shape(test_data)[0])
    
    bgdata = np.array(minbias_filterandpad(minbias_pu200, maxhits, minbiasvars, interested_vars))
    
    test_labels = torch.cat((test_labels, torch.zeros(np.shape(bgdata)[0])))
    
    test_data = np.concatenate((test_data, bgdata))
    test_data = torch.tensor(test_data)
    
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    classifier = Net(maxhits, args)
    
    if args.early_stop == True:
        classifier.load_state_dict(torch.load("{0}/BestClassifierSettings".format(path)))
    else:     
        classifier.load_state_dict(torch.load("{0}/FinalClassifierSettings".format(path)))
        
    classifier.eval()
        
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_data), shuffle=True, drop_last=True)
    
    pu0data = np.array(filterandpad(pu0data, maxhits, taugunvars, interested_vars, args.extra_filter))
    pu200data = np.array(filterandpad(pu200data, maxhits, taugunvars, interested_vars, args.extra_filter))
    
    pu0data = torch.tensor(pu0data)
    pu200data = torch.tensor(pu200data)
    bgdata = torch.tensor(bgdata)
    
    testroc(classifier, test_data_loader, path, args)
    plotAccuracy(classifier, pu0data, pu200data, bgdata, path, args)
        
        
        
        
        
        
        
        
        
        
        
        
        
