import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from utils2 import *
from test_model1_4 import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as hp

def train_one_batch(classifier, optimizer, train_data, labels):
    torch.set_default_dtype(torch.float32)
    torch.enable_grad()
    # Initialize our loss function
    loss = FocalLoss(.5, 3)
    # Reset the optimizer
    optimizer.zero_grad()
    
    # Make a prediction from a batch
    prediction = torch.flatten(classifier(train_data)).float()
    labels.float()
    
    # Get error from the model's prediction
    error = loss(prediction, labels)
    
    # Propagate the error
    error.backward()
    
    # Update weights and biases using the optimizer
    optimizer.step()
    
    return error, prediction
    
def trainAUC(classifier, auc_loader, path, epoch):
    classifier.eval()
    
    max_fpr = .001
    
    R_LHC = 2760*11.246 ##kHz
    
    # Generates an auc curve using the training dataset
    # This loader only has one batch, this "loop" is for ease of use
    for batch, labels in auc_loader:
        with torch.no_grad():
            prediction = torch.flatten(classifier(batch))
            fpr, tpr, thresholds = metrics.roc_curve(labels, prediction)
            roc_auc = metrics.auc(fpr, tpr)
            partial_roc_auc = metrics.roc_auc_score(labels, prediction, max_fpr=max_fpr)
    
    if epoch%10 == 0:
        
        index = get_idx_for_interested_fpr(fpr, max_fpr)
        interested_tpr = float(tpr[index])
    
        lw = 2
        plt.clf()
        plt.plot(fpr*R_LHC, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.ylim([0.0, 1.05])
        plt.xlim((0, 100))
        plt.axvline(x = 30, label="30 kHz", linestyle='dashed')
        #plt.plot(max_fpr*R_LHC, interested_tpr, 'ro', label = '.001 Fpr, %.2f Recall (area = %.4f)' % (interested_tpr, partial_roc_auc))
        plt.xlabel('Trigger Rate (kHz)')
        plt.ylabel('Trigger Acceptance')
        plt.title('Epoch #{0} Training ROC Curve'.format(epoch))
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("{0}/Training_ROC_Curve_Epoch{1}.png".format(path, epoch))            
        plt.close()
    
    return roc_auc

    
