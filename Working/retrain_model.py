import argparse
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from utils import *
from train import train_one_batch
from datetime import datetime
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
from tensorflow.summary import create_file_writer, scalar

class hyperTunnel(nn.Module):
    def __init__(self, maxhits, num_hidden, nodes_per_hidden, dropout=0.0):
        super(hyperTunnel, self).__init__()
        
        self.input_size = 8*maxhits
        self.num_hidden=num_hidden
        self.dropout = dropout
        
        self.batch_norm = nn.BatchNorm1d(self.input_size)
        
        self.input_layer = nn.Sequential(
                                nn.Linear(self.input_size, nodes_per_hidden),
                                nn.LeakyReLU(),
                                nn.Dropout(self.dropout)
                            )
        
        self.hidden_layers = nn.ModuleList()
        self.nodes_per_hidden = nodes_per_hidden
        
        for num in range(num_hidden):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(nodes_per_hidden, nodes_per_hidden),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout)
                    )
                )

        self.output_layer = nn.Sequential(
                                nn.Linear(nodes_per_hidden, 1),
                                nn.Sigmoid()
                            )
            
    def forward(self, x):
        x = x.float()
        x = self.batch_norm(x)
        #x.cuda()
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        return x

def plotAccuracy(classifier, valid_plot_loader, train_plot_loader, path, epoch):
    
    if epoch%50 == 0:
        
        valid_signal_preds = []
        valid_bg_preds = []
        train_signal_preds = []
        train_bg_preds = []
        
        classifier.eval()
        for pu200_event, label in valid_plot_loader:
            with torch.no_grad():
                if label == 1:
                    valid_signal_preds.append(float(classifier(pu200_event)))
                
                if label == 0:
                    valid_bg_preds.append(float(classifier(pu200_event)))

        for pu200_event, label in train_plot_loader:
            with torch.no_grad():
                if label == 1:
                    train_signal_preds.append(float(classifier(pu200_event)))
                
                if label == 0:
                    train_bg_preds.append(float(classifier(pu200_event)))
        
        plt.hist(valid_signal_preds, histtype='step',bins=int(len(valid_signal_preds)**.5), label="Signal pu200", density=True)
        plt.hist(valid_bg_preds, histtype='step',bins=int(len(valid_bg_preds)**.5), label="Background pu200", density=True)
        plt.xticks(ticks = [.1*i for i in range(11)])
        plt.title("Predictions")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend(loc="upper center")
        plt.savefig("{0}/Epoch{1}_ValidAccuracyPlot.png".format(path, epoch))
        plt.clf()
        
        plt.hist(train_signal_preds, histtype='step',bins=int(len(train_signal_preds)**.5), label="Signal pu200", density=True)
        plt.hist(train_bg_preds, histtype='step',bins=int(len(train_bg_preds)**.5), label="Background pu200", density=True)
        plt.xticks(ticks = [.1*i for i in range(11)])
        plt.title("Predictions")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend(loc="upper center")
        plt.savefig("{0}/Epoch{1}_TrainAccuracyPlot.png".format(path, epoch))

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
    
    if epoch%10 == 0 or epoch == 1:
        
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
        plt.savefig("{0}/Train_Epoch{1}.png".format(path, epoch))            
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
    
    R_LHC = 2760*11.246

    if epoch%10 == 0 or epoch == 1:
        print(tpr)
        index = get_idx_for_interested_fpr(fpr, max_fpr)
        print(tpr[index])
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
        plt.savefig("{0}/Valid_Epoch{1}.png".format(path, epoch))    
        
        torch.save(classifier.state_dict(), "{0}/Settings_Epoch{1}".format(path,epoch))
        
        plt.close()
        
    return error, roc_auc


def trainModel(model, lr):
    batch_size = 64
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    args = None
    
    hparams = {
            HP_HIDDEN : 5,
            HP_LR : lr,
            HP_DROPOUT : 0.444
        }
        
    path = "retrain_runs/{0}H_{1}LR_{2}B_{3}DR_{4}".format(model.num_hidden, round(lr, 6), batch_size, round(model.dropout, 3), datetime.now().strftime('%b%d_%H-%M-%S'))
    
    epochs = 150
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_auc_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_data), shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_data), shuffle=True, drop_last=True)
    
    tb = SummaryWriter(log_dir=path)
    
    model.float()
    for epoch in range(1, epochs+1):
        errors = []
        
        for batch, labels in train_loader:
            error, prediction = train_one_batch(model, opt, batch, labels)
            errors.append(float(error))
        
    
        valid_error, valid_auc_score = validate(model, valid_loader, epoch, path, args)
        train_auc_score = trainAUC(model, train_auc_loader, path, epoch)
        tb.add_scalar("Training Loss", np.mean(errors), epoch)
        tb.add_scalar("Training AUROC", float(train_auc_score), epoch)
        tb.add_scalar("Validation Loss", float(valid_error), epoch)
        tb.add_scalar("Validation AUROC", float(valid_auc_score), epoch)        
        tb.add_histogram('Input Layer Weights', model.input_layer[0].weight, epoch)
        
        plotAccuracy(model, valid_plot_loader, train_plot_loader, path, epoch)
        
    with create_file_writer(tb.log_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        scalar('accuracy', valid_auc_score, step=1)  
    
    tb.flush()
    
    return valid_auc_score

def main():

    global HP_HIDDEN
    global HP_LR
    global HP_BATCH
    global HP_DROPOUT
    global METRIC_ACCURACY
    
    torch.set_default_dtype(torch.float32)
    
    
    HP_HIDDEN = hp.HParam('# hidden layers', hp.Discrete([i for i in range(3,9)]))
    HP_LR = hp.HParam('learning rate', hp.RealInterval(.0001, .0009))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.5))
    METRIC_ACCURACY = 'accuracy'
    
    with create_file_writer('retrain_runs/hparam_tuning').as_default():
      hp.hparams_config(
        hparams=[HP_HIDDEN, HP_LR, HP_DROPOUT],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Val AUROC')]
      )
    
    global maxhits
    maxhits = 50
    interested_vars = ['mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_theta', 'mu_hit_sim_z', 'mu_hit_bend', 'mu_hit_ring', 'mu_hit_quality']
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    
    dev = "cpu"
    device = torch.device(dev)
    
    MinBias = open("MinBiasPU200_MTD.pkl", "rb")
    minbias_pu200file = pickle.load(MinBias)
    MinBias.close()
    minbiasvars = minbias_pu200file.keys()
    minbias_pu200 = np.array(minbias_pu200file)
    del minbias_pu200file        
    
    PU200_MTD = open("DsTau3muPU200_MTD.pkl", "rb")
    tau_pu200file = pickle.load(PU200_MTD)
    taugunvars = tau_pu200file.keys()
    PU200_MTD.close()
    pu200data = np.array(tau_pu200file)
    del tau_pu200file
    
    global train_data
    global valid_data
    
    signal_data, indices = filterandpad(pu200data, maxhits, taugunvars, interested_vars, 0, one_endcap=True)
    bgdata = np.array(minbias_filterandpad(minbias_pu200,maxhits, minbiasvars, interested_vars, one_endcap=True))
    
    print(signal_data.shape)
    print(bgdata.shape)
    
    np.random.seed(17)
    shuffler = np.random.permutation(len(signal_data))
    signal_data[shuffler]
    
    np.random.seed(17)
    bgshuffler = np.random.permutation(len(bgdata))
    bgdata[bgshuffler]
    
    train_data, valid_data = np.split(signal_data, [int(.95*np.shape(signal_data)[0])])
    
    bgtrain, bgvalid = np.split(bgdata, [int(4.92*np.shape(train_data)[0])])
   
    del signal_data
    del bgdata
    
    train_labels = torch.ones(np.shape(train_data)[0])
    valid_labels = torch.ones(np.shape(valid_data)[0])
    
    train_labels = torch.cat((train_labels, torch.zeros(np.shape(bgtrain)[0])))
    valid_labels = torch.cat((valid_labels, torch.zeros(np.shape(bgvalid)[0])))
    print(np.shape(valid_data))
    print(np.shape(bgtrain))
    print(np.shape(bgvalid))
    print(train_labels)
    print(valid_labels)

    train_data = np.concatenate((train_data, bgtrain))
    valid_data = np.concatenate((valid_data, bgvalid))
    
    train_data = torch.tensor(train_data, dtype=torch.double)
    valid_data = torch.tensor(valid_data, dtype=torch.double)
    
    global train_dataset
    global valid_dataset
    
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_labels)

    global train_plot_loader
    global valid_plot_loader
    
    train_plot_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    valid_plot_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=True)
    
    model = hyperTunnel(maxhits, 5, 256, dropout=.444)
    lr = .000686
    
    trainModel(model, lr)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
