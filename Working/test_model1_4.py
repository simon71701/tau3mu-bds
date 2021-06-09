import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import *
from train import *

def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')

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


def testModel(classifier, test_data_loader, path, args):
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
    
    
def main():
    
    args = arg_parse()
    print(args.test_on)
    
    interested_vars = ['mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r']
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    
    path = "TrainingLogs/"
    
    if args.extra_filter:
        path += "Extra_Filter/"
    else:
        path += "No_Extra_Filter/"
    
    if args.mix:
        path += "{0}Mix/".format(int(args.mix*100))
    else:
        path += "No_Mix/pu{0}/".format(args.pileup)
        
    path += "{0}Layers_{1}Dropout".format(args.num_funnel_layers, args.dropout)
    
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
        classifier.load_state_dict(torch.load("{0}/Epoch200_ClassifierSettings".format(path)))
        
    classifier.eval()
        
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_data), shuffle=True, drop_last=True)
    
    
    pu0data = np.array(filterandpad(pu0data, maxhits, taugunvars, interested_vars, args.extra_filter))
    pu200data = np.array(filterandpad(pu200data, maxhits, taugunvars, interested_vars, args.extra_filter))
    
    pu0data = torch.tensor(pu0data)
    pu200data = torch.tensor(pu200data)
    bgdata = torch.tensor(bgdata)
    
    testModel(classifier, test_data_loader, path, args)
    plotAccuracy(classifier, pu0data, pu200data, bgdata, path, args)

if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
