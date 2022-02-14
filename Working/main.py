import argparse
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from utils3 import *
from datetime import datetime
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Define the neural network using Torch.
class Classifier(nn.Module):
    def __init__(self, maxhits, num_hidden, nodes_per_hidden, dropout=0.0):
        super(Classifier, self).__init__()
        
        # Set the size of the network's input
        self.input_size = 8*maxhits
        # Set the desired number of hidden layers
        self.num_hidden=num_hidden
        # Set the dropout for each layer. Dropout is the probability that a node will be killed off during training. Simplifies the model and prevents overtraining
        self.dropout = dropout
        
        # Define the batch_norm layer. Normalizes the input batch
        self.batch_norm = nn.BatchNorm1d(self.input_size)
        
        # Define the input layer.
        self.input_layer = nn.Sequential(
                                nn.Linear(self.input_size, nodes_per_hidden),
                                nn.LeakyReLU(),
                                nn.Dropout(self.dropout)
                            )
        
        # Initialize the hidden layers
        self.hidden_layers = nn.ModuleList()
        # Set how many nodes should be in each hidden layer
        self.nodes_per_hidden = nodes_per_hidden
        
        # Fill up the hidden layers
        for num in range(num_hidden):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(nodes_per_hidden, nodes_per_hidden),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout)
                    )
                )
        
        # Define the output layer. The Sigmoid function forces the input to be between 0 and 1.
        self.output_layer = nn.Sequential(
                                nn.Linear(nodes_per_hidden, 1),
                                nn.Sigmoid()
                            )
        
    # Define how the network treats the input 
    def forward(self, x):
        x = x.float()
        x = self.batch_norm(x)
        x.cuda()
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        
        return x

# Plots the distribution of predictions for events in both the training dataset and the validation dataset
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

# Plots the model's AUC curve for the training dataset every 10th epoch 
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
        plt.axvline(x = 30, label="30 kHz \n(Acceptance = {0}".format(round(interested_tpr,4)), linestyle='dashed')
        #plt.plot(max_fpr*R_LHC, interested_tpr, 'ro', label = '.001 Fpr, %.2f Recall (area = %.4f)' % (interested_tpr, partial_roc_auc))
        plt.xlabel('Trigger Rate (kHz)')
        plt.ylabel('Trigger Acceptance')
        plt.title('Epoch #{0} Training ROC Curve'.format(epoch))
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig("{0}/Train_Epoch{1}.png".format(path, epoch))            
        plt.close()
    
    return roc_auc

# Performs model validation, then plots the AUC curve for it.
def validate(classifier, valid_data_loader, epoch, path, args, early_stop=False):
    
    classifier.eval()
    loss = FocalLoss(.5, 3)
    errors = []
    
    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    
    max_fpr = .001
    
    # There's only one batch in valid_data_loader, this just ensures that the loader outputs the batch in the correct way.
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
        plt.legend(loc="upper right")
        plt.grid()
        plt.savefig("{0}/Valid_Epoch{1}.png".format(path, epoch))    
        
        torch.save(classifier.state_dict(), "{0}/Settings_Epoch{1}".format(path,epoch))
        
        plt.close()
        
    return error, roc_auc

# Performs model training.
def trainModel(model, lr):
    batch_size = 64
    # Create the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    args = None
    
     
    # Set the path for function outputs, ie plots and settings
    current_dir = os.getcwd()
    retrain_dir = os.path.join(current_dir, "retrain_runs")
    comment=''
    new_dir = "{0}H_{1}_Nodes_{2}LR_{3}B_{4}DR_{5}_{6}".format(model.num_hidden, model.nodes_per_hidden, round(lr, 6), batch_size, round(model.dropout, 3), datetime.now().strftime('%b%d_%H-%M-%S'), comment)
    path = os.path.join(retrain_dir, new_dir)
    os.mkdir(path)
    path = str(path)
    epochs = 150
    
    writer = SummaryWriter(log_dir=path)
    
    # Initialize data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_auc_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True, drop_last=True)
    
    model.float()
    
    # Training loop
    for epoch in tqdm(range(1, epochs+1)):
        errors = []
        
        # Train and get errors from each batch
        for batch, labels in train_loader:
            error, prediction = train_one_batch(model, opt, batch, labels)
            errors.append(float(error))
        
        # Validate the model
        valid_error, valid_auc_score = validate(model, valid_loader, epoch, path, args) 
     
        train_auc_score = trainAUC(model, train_auc_loader, path, epoch)
        
        writer.add_scalar('Loss/Train', np.mean(errors), epoch)
        writer.add_scalar('Loss/Validation', valid_error, epoch)
        
        writer.add_scalar('AUROC/Train', train_auc_score, epoch)
        writer.add_scalar('AUROC/Validation', valid_auc_score, epoch)
        
        writer.flush()
   
        # Create accuracy plots
        plotAccuracy(model, valid_plot_loader, train_plot_loader, path, epoch)
    
    return valid_auc_score

def main():
  
    torch.set_default_dtype(torch.float32)
    
    global maxhits
    
    # The main drawback of using FCNs is that we have to make a choice of data size since all the input vectors have to be the same size. We set the maximum hits per event
    # to limit the size of the input while still using at least 99% of the data.
    maxhits = 50
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 


    device = torch.device(dev)
    
    signal_train = np.load('/scratch/gilbreth/simon73/Working/SignalPU200_Train.npy')
    signal_valid = np.load('/scratch/gilbreth/simon73/Working/SignalPU200_Valid.npy')
    bg_train = np.load('/scratch/gilbreth/simon73/Working/BgPU200_Train.npy')
    bg_valid = np.load('/scratch/gilbreth/simon73/Working/BgPU200_Valid.npy')
    
    # Create the labels for the signal events
    train_labels = torch.ones(np.shape(signal_train)[0])
    valid_labels = torch.ones(np.shape(signal_valid)[0])
    
    # Add on the labels for the background events
    train_labels = torch.cat((train_labels, torch.zeros(np.shape(bg_train)[0])))
    valid_labels = torch.cat((valid_labels, torch.zeros(np.shape(bg_valid)[0])))
    
    # Compile final datasets
    train_data = np.concatenate((signal_train, bg_train))
    valid_data = np.concatenate((signal_valid, bg_valid))
    
    # Convert to tensors for torch
    train_data = torch.tensor(train_data, dtype=torch.double)
    valid_data = torch.tensor(valid_data, dtype=torch.double)
    
    global train_dataset
    global valid_dataset
    
    # Create Tensor Datasets for ease of use with data loaders
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_labels)

    global train_plot_loader
    global valid_plot_loader
    
    # Create data loaders for the plotting functions
    train_plot_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    valid_plot_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, drop_last=True)
    
    # Initialize the desired model
    model = Classifier(maxhits, 5, 256, dropout=.444)
    # Initialize the desired learning rate
    lr = .000686
    
    # Train the model
    trainModel(model, lr)

# This ensures that running the file in the terminal will actually run the code
if __name__ == '__main__':
    main()
