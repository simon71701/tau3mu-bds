import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from utils import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    
    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
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
        plt.title('ROC Curve: {0} Trainable Parameters'.format(total_params))
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
        plt.title('Epoch #{0} ROC Curve: {1} Trainable Parameters'.format(epoch, total_params))
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("{0}/ROC_Curve_Epoch{1}.png".format(path, epoch))            
        plt.close()
        
    return error, roc_auc

def trainModel(train_data, test_data, maxhits, device, args):
    
    auc_score = .50
    running = True
    retries = 0
    
    
    while auc_score == .50 and running == True:
        classifier = Net(maxhits, args)
        classifier.to(device)
        opt = torch.optim.Adam(classifier.parameters(),lr=args.lr)
        
        if args.extra_filter == 1:
            comment = '_{0}Pileup_{1}FL_{2}Dropout_{3}Mix_ExtraFilter'.format(args.pileup, args.num_funnel_layers, args.dropout, args.mix)
        else:
            comment = '_{0}Pileup_{1}FL_{2}Dropout_{3}Mix_NoExtraFilter'.format(args.pileup, args.num_funnel_layers, args.dropout, args.mix)
        
        tb = SummaryWriter(comment=comment)
        
        if args.early_stop:
            early_stopping = EarlyStopping(path="{0}/checkpoint.pt".format(tb.log_dir), patience=10)
        
        
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, drop_last=True)
        
        events, labels = next(iter(data_loader))
        grid = torchvision.utils.make_grid(events)
        tb.add_graph(classifier, events)
        
        for epoch in tqdm(range(1, args.epochs+1)):
            errors = []
            for batch, labels in data_loader:
                error, prediction = train_one_batch(classifier, opt, batch, labels)
                errors.append(float(error))
            
            if args.early_stop:
                test_error, auc_score = test(classifier, test_data_loader, epoch, tb.log_dir, args, early_stop=early_stopping.early_stop)
            else:
                test_error, auc_score = test(classifier, test_data_loader, epoch, tb.log_dir, args)
            
            if epoch== 1 and auc_score == .50:
                print("Retrying")
                retries += 1
                del classifier
                del opt
                del data_loader
                del test_data_loader
                break
        
            tb.add_scalar("Training Loss", np.mean(errors), epoch)
            tb.add_scalar("Validation Loss", float(test_error), epoch)
            tb.add_scalar("Validation AUROC", float(auc_score), epoch)
            
            tb.add_histogram('funnel_layers.1.0.weight', classifier.funnel_layers[1][0].weight, epoch)
            tb.add_histogram('funnel_layers.1.0.bias', classifier.funnel_layers[1][0].bias, epoch)
            tb.add_histogram(
                'funnel_layers.1.0.weight.grad'
                ,classifier.funnel_layers[1][0].weight.grad
                ,epoch
            )
            
            if args.early_stop:
                early_stopping(test_error, classifier)
            
                if early_stopping.early_stop:
                    print("Early stopping")
                    test_error = test(classifier, test_data_loader, epoch-10, tb.log_dir, args, early_stop=early_stopping.early_stop)
                    classifier.load_state_dict(torch.load('{0}/checkpoint.pt'.format(tb.log_dir)))
                    torch.save(classifier.state_dict(), "{0}/BestClassifierSettings".format(tb.log_dir))
                    
                    stopping_point = epoch-10
                    
                    break
                    
            if epoch%10 == 0 and epoch != args.epochs:
                torch.save(classifier.state_dict(), "{0}/Epoch{1}_ClassifierSettings".format(tb.log_dir, epoch))
            
            if epoch == args.epochs:
                running == False
                torch.save(classifier.state_dict(), "{0}/FinalClassifierSettings".format(tb.log_dir, epoch))
                
                tb.close()
    
    return tb.log_dir
    
