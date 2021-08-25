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
    loss = FocalLoss(.5, 3)
    optimizer.zero_grad()
    
    prediction = torch.flatten(classifier(train_data)).float()
    labels.float()
    error = loss(prediction, labels)
    error.backward()
    
    optimizer.step()
    
    return error, prediction
    
def trainAUC(classifier, auc_loader, path, epoch):
    classifier.eval()
    
    max_fpr = .001
    
    R_LHC = 2760*11.246 ##kHz
    
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

def validate(classifier, valid_data_loader, epoch, path, args, early_stop=False):
    
    classifier.eval()
    loss = FocalLoss(.5, 3)
    errors = []
    
    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
    max_fpr = .001
    
    for batch, labels in valid_data_loader:
        with torch.no_grad():
            prediction = torch.flatten(classifier(batch))
            error = loss(prediction, labels)
            fpr, tpr, thresholds = metrics.roc_curve(labels, prediction)
            roc_auc = metrics.auc(fpr, tpr)
            partial_roc_auc = metrics.roc_auc_score(labels, prediction, max_fpr=max_fpr)
    
    R_LHC = 2760*11.246
    if early_stop == True:
        index = get_idx_for_interested_fpr(fpr, max_fpr)
        interested_tpr = float(tpr[index])   
        fpr = fpr*R_LHC
        lw = 2
        plt.clf()
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.ylim([0.0, 1.05])
        plt.xlim((0, 100))
        #plt.plot(, interested_tpr, 'ro', label = '.001 Fpr, %.2f Recall (area = %.4f)' % (interested_tpr, partial_roc_auc))
        plt.xlabel('Trigger Rate (kHz)')
        plt.ylabel('Trigger Acceptance')
        plt.title('ROC Curve: {0} Trainable Parameters'.format(total_params))
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("{0}/Best_Validation_ROC_Curve.png".format(path))            
        plt.close()
        
    elif epoch%10 == 0:
        index = get_idx_for_interested_fpr(fpr, max_fpr)
        interested_tpr = float(tpr[index])  
        fpr = fpr*R_LHC
        lw = 2
        plt.clf()
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.xlim((0, 100))
        plt.ylim([0.0, 1.05])
        plt.axvline(x = 30, label="30 kHz", linestyle='dashed')
        #plt.plot(max_fpr, interested_tpr, 'ro', label = '.001 Fpr, %.2f Recall (area = %.4f)' % (interested_tpr, partial_roc_auc))
        plt.xlabel('Trigger Rate (kHz)')
        plt.ylabel('Trigger Acceptance')
        plt.title('Epoch #{0} ROC Curve: {1} Trainable Parameters'.format(epoch, total_params))
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("{0}/Validation_ROC_Curve_Epoch{1}.png".format(path, epoch))            
        plt.close()
        
    return error, roc_auc

def trainModel(train_data, valid_data, test_data, maxhits, device, args):
    
    valid_auc_score = .50
    running = True
    retries = 0
    
    
    while valid_auc_score == .50 and running == True:
        classifier = Net(maxhits, args)
        #classifier.to(device)
        opt = torch.optim.Adam(classifier.parameters(),lr=args.lr)
        
        if args.extra_filter == 1:
            comment = '_{0}Pileup_{1}FL_{2}Dropout_{3}Mix_ExtraFilter'.format(args.pileup, args.num_funnel_layers, args.dropout, args.mix)
        else:
            comment = '_{0}Pileup_{1}FL_{2}Dropout_{3}Mix_NoExtraFilter'.format(args.pileup, args.num_funnel_layers, args.dropout, args.mix)
        
        tb = SummaryWriter(comment=comment)
        
        if args.early_stop:
            early_stopping = EarlyStopping(path="{0}/checkpoint.pt".format(tb.log_dir), patience=10)
        
        
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=len(valid_data), shuffle=True, drop_last=True)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True, drop_last=True)
        
        events, labels = next(iter(data_loader))
        
        for epoch in tqdm(range(1, args.epochs+1)):
            errors = []
            for batch, labels in data_loader:
                error, prediction = train_one_batch(classifier, opt, batch, labels)
                errors.append(float(error))
            
            if args.early_stop:
                valid_error, valid_auc_score = validate(classifier, valid_data_loader, epoch, tb.log_dir, args, early_stop=early_stopping.early_stop)
            else:
                valid_error, valid_auc_score = validate(classifier, valid_data_loader, epoch, tb.log_dir, args)
                test_error, test_auc_score = testModel(classifier, test_data_loader, epoch, tb.log_dir, args)
            
            if epoch== 1 and valid_auc_score == .50:
                print("Retrying")
                retries += 1
                del classifier
                del opt
                del data_loader
                del valid_data_loader
                del test_data_loader
                break
        
            tb.add_scalar("Training Loss", np.mean(errors), epoch)
            tb.add_scalar("Validation Loss", float(valid_error), epoch)
            tb.add_scalar("Validation AUROC", float(valid_auc_score), epoch)
            tb.add_scalar("Testing Loss", float(test_error), epoch)
            tb.add_scalar("Testing AUROC", float(test_auc_score), epoch)
            
            tb.add_histogram('funnel_layers.1.0.weight', classifier.funnel_layers[1][0].weight, epoch)
            tb.add_histogram('funnel_layers.1.0.bias', classifier.funnel_layers[1][0].bias, epoch)
            tb.add_histogram(
                'funnel_layers.1.0.weight.grad'
                ,classifier.funnel_layers[1][0].weight.grad
                ,epoch
            )
            
            if args.early_stop:
                early_stopping(valid_error, classifier)
            
                if early_stopping.early_stop:
                    print("Early stopping")
                    valid_error = validatet(classifier, valid_data_loader, epoch-10, tb.log_dir, args, early_stop=early_stopping.early_stop)
                    classifier.load_state_dict(torch.load('{0}/checkpoint.pt'.format(tb.log_dir)))
                    torch.save(classifier.state_dict(), "{0}/BestClassifierSettings".format(tb.log_dir))
                    
                    stopping_point = epoch-10
                    
                    break
                    
            if epoch%10 == 0 and epoch != args.epochs:
                torch.save(classifier.state_dict(), "{0}/Epoch{1}_ClassifierSettings".format(tb.log_dir, epoch))
            
            if epoch == args.epochs:
                running = False
                torch.save(classifier.state_dict(), "{0}/FinalClassifierSettings".format(tb.log_dir, epoch))
                testModel(classifier, test_data_loader, epoch, tb.log_dir, args, running=False)
                
                tb.close()
    
