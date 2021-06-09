import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self, maxhits, args):
        super(Net, self).__init__()
        
        self.batch_size = 3*args.maxhits
        self.num_funnel_layers=args.num_funnel_layers
        self.funnel_rate = (self.batch_size-3)/(self.num_funnel_layers+1)
        self.dropout=args.dropout
        
        self.funnel_layers = nn.ModuleList()
        self.funnel_layers.append(nn.BatchNorm1d(self.batch_size))
        
        for i in range(self.num_funnel_layers):
            self.funnel_layers.append(
                
            nn.Sequential( 
                nn.Linear(int(self.batch_size-self.funnel_rate*i), int(self.batch_size-self.funnel_rate*(i+1))),
                nn.ReLU(),
                nn.Dropout(self.dropout)
                )
                
            )
        
        self.funnel_layers.append(
            
            nn.Sequential(
                nn.Linear(int(self.batch_size-self.funnel_rate*(self.num_funnel_layers)), 3),
                nn.ReLU(),
                nn.Dropout(self.dropout)
                )
            
            )
        
        self.output_layer = nn.Sequential( 
            nn.Linear(3, 1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        for layer in self.funnel_layers:
            x = layer(x)
            
        x = self.output_layer(x)
        return x

def train_one_batch(classifier, optimizer, train_data, labels):
    torch.enable_grad()
    loss = FocalLoss(.5, 3)
    optimizer.zero_grad()
    
    prediction = torch.flatten(classifier(train_data))
    
    error = loss(prediction, labels)
    error.backward()
    
    optimizer.step()
    
    return error, prediction

def test(classifier, test_data_loader, epoch, path, args, early_stop=False):
    
    classifier.eval()
    loss = FocalLoss(.5, 3)
    errors = []
    
    max_fpr = .001
    
    for batch, labels in test_data_loader:
        with torch.no_grad():
            prediction = torch.flatten(classifier(batch))
            error = loss(prediction, labels)
            fpr, tpr, thresholds = metrics.roc_curve(labels, prediction)
            roc_auc = metrics.auc(fpr, tpr)
            partial_roc_auc = metrics.roc_auc_score(labels, prediction, max_fpr=max_fpr)
    
    index = get_idx_for_interested_fpr(fpr, .01)
    interested_tpr = float(tpr[index])
    
    if early_stop == True:
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
        plt.title('ROC Curve'.format(epoch))
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("{0}/Best_ROC_Curve.png".format(path))            
        plt.close()
        
    elif epoch%10 == 0:
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
        plt.title('ROC Curve: Epoch #{0}'.format(epoch))
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("{0}/ROC_Curve_Epoch{1}.png".format(path, epoch))            
        plt.close()
        
    return error, roc_auc

def trainModel(train_data, test_data, maxhits, device, args):
    
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
    
    if args.early_stop:
        early_stopping = EarlyStopping(path="{0}/checkpoint.pt".format(path), patience=10)
    
    auc_score = .50
    running = True
    retries = 0
    
    
    while auc_score == .50 and running == True:
        classifier = Net(maxhits, args)
        classifier.to(device)
        opt = torch.optim.Adam(classifier.parameters(),lr=args.lr)
        
        total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, drop_last=True)
        
        epoch_errors = []
        
        test_errors = []
        
        for epoch in tqdm(range(1, args.epochs+1)):
            errors = []
            for batch, labels in data_loader:
                error, prediction = train_one_batch(classifier, opt, batch, labels)
                errors.append(float(error))
            
            epoch_errors.append(np.mean(errors))
            
            if args.early_stop:
                test_error, auc_score = test(classifier, test_data_loader, epoch, path, args, early_stop=early_stopping.early_stop)
            else:
                test_error, auc_score = test(classifier, test_data_loader, epoch, path, args)
            
            if epoch== 1 and auc_score == .50:
                print("Retrying")
                retries += 1
                del classifier
                del opt
                del data_loader
                del test_data_loader
                break
        
            test_errors.append(float(test_error))
            
            if args.early_stop:
                early_stopping(test_error, classifier)
            
                if early_stopping.early_stop:
                    print("Early stopping")
                    test_error = test(classifier, test_data_loader, epoch-10, path, args, early_stop=early_stopping.early_stop)
                    classifier.load_state_dict(torch.load('{0}/checkpoint.pt'.format(path)))
                    torch.save(classifier.state_dict(), "{0}/BestClassifierSettings".format(path))
                    
                    stopping_point = epoch-10
                    
                    break
            
            if epoch%10 == 0:
                print("Epoch {0} Finished: Current Total Training Error={1} ".format(epoch+1, error))
                
                plt.clf()                                   
                plt.plot(range(0,len(list(epoch_errors))), list(epoch_errors))
                plt.xlabel("Epoch")
                plt.title("Training Error: FL={0}, {1} Trainable Parameters".format(args.num_funnel_layers, total_params))
                plt.ylabel("Total Training Error")
                plt.grid()
                plt.savefig("{0}/FunnelErrorCurve_Epoch={1}.png".format(path, epoch))
                plt.clf()
                
            
                if epoch != 0:
                    test_epochs = [i for i in range(len(test_errors))]
                    plt.plot(test_epochs, list(test_errors))
                    plt.xlabel("Epoch")
                    plt.title("Testing Error: FL={0}".format(args.num_funnel_layers))
                    plt.ylabel("Total Test Error")
                    plt.grid()
                    plt.savefig("{0}/FunnelErrorCurve_Test_Epoch={1}.png".format(path, epoch))
                    plt.clf()
                
                torch.save(classifier.state_dict(), "{0}/Epoch{1}_ClassifierSettings".format(path, epoch))
            
            if epoch == args.epochs:
                running == False
        
            
    
