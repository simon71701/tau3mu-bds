import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

## Optimizing Hyperparameters:
##  batch_size
##  num_funnel_layers
##  dropout
##  Learning Rate


class Net(nn.Module):
    def __init__(self, maxhits, args):
        super(Net, self).__init__()
        
        self.batch_size = 3*args.maxhits
        self.num_funnel_layers=args.num_funnel_layers
        self.funnel_rate = (self.batch_size-3)/(self.num_funnel_layers+1)
        self.dropout=args.dropout
        
        self.funnel_layers = nn.ModuleList()
        
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


    parser.set_defaults(batch_size=10,
                        dropout=.05,
                        epochs=150,
                        lr=.001,
                        pileup=0,
                        num_funnel_layers=5,
                        maxhits=50)

    return parser.parse_args()

## Filter for events that only have 1 tau
def filterPU200(dataset):
    singledOut = []
    
    for event in dataset:
        if event[0] == 1:
            singledOut.append(event)
    
    return singledOut

## Return a filtered dataset that only has the specified characteristic
def collectChar(index, dataset):
    char = []
    
    for i in range(len(dataset)):
        char.append(dataset[i][index])
    
    
    return char

# Filter for events that have up to a specified number of muons and pad events with zeros.  Ordered (Phi, Eta, R) 
def filterandpad(dataset, maxhits):
    filtered = []
    for event in dataset:
        chars = []
        #print(event[2])
        #print(maxhits)
        if event[2] <= maxhits:
            for i in (29, 31, 32):
                char = np.pad(event[i], (0,maxhits-len(event[i])), 'constant', constant_values=0)
                chars.append(char)
            filtered.append(chars)
    return filtered

# See filterandpad
def minbias_filterandpad(minbias,maxhits):
    filtered = []
    for event in minbias:
        chars = []
        if event[0] <= maxhits:
            for i in (17, 19, 20):
                char = np.pad(event[i], (0,maxhits-len(event[i])), 'constant', constant_values=0)
                chars.append(char)
            filtered.append(chars)
    
    return filtered     

def noise(batch_size, maxhits):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = torch.randn(batch_size, 3*maxhits)
    return n

def ones_target(batch_size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = torch.ones(batch_size, 1)
    return data

def zeros_target(batch_size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(batch_size, 1)
    return data

def train_one_batch(classifier, optimizer, real_data, fake_data_loader, args):
    loss = nn.BCELoss()
    optimizer.zero_grad()
    
    real_data = torch.flatten(real_data, start_dim=1)
    
    prediction_real = classifier(real_data)
    error_real = loss(prediction_real, ones_target(args.batch_size))
    error_real.backward()
    
    for batchnum, batch in enumerate(fake_data_loader):
        fake_data = torch.flatten(batch, start_dim=1)
        prediction_fake = classifier(fake_data)
        error_fake = loss(prediction_fake, zeros_target(args.batch_size))
        error_fake.backward()
        break
    
    optimizer.step()
    
    return error_real + error_fake, prediction_real, prediction_fake

def trainModel(dataset, bgdataset, maxhits, device, args):
    classifier = Net(maxhits, args)
    classifier.to(device)
    
    dataset = torch.tensor(dataset)
    dataset.to(device)
    bgdataset = torch.tensor(bgdataset)
    bgdataset.to(device)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    fake_data_loader = torch.utils.data.DataLoader(bgdataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    errors = []
    opt = torch.optim.Adam(classifier.parameters(),lr=.001)
    
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for batchnum, batch in enumerate(data_loader):
            batch.to(device)
            error, real, fake = train_one_batch(classifier, opt, batch, fake_data_loader, args)
            
        print("Epoch {0} Finished: Total Error={1} ".format(epoch+1, error))
        print("Prediction on Real Data: ",real)
        print("Prediction on Background Data: ",fake)
        errors.append(float(error))
                                         
    plt.plot(range(1,args.epochs+1), list(errors))
    plt.xlabel("Epoch")
    plt.title("Error Curve")
    plt.ylabel("Total Error: FL={0}".format(args.num_funnel_layers))
    plt.savefig("FunnelErrorCurve_fl={0}.png".format(args.num_funnel_layers))

def main():
    
    args = arg_parse()
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    
    device = torch.device(dev)
    
    MinBias = open("MinBiasPU200_MTD.pkl", "rb")
    minbias_pu200file = pickle.load(MinBias)
    minbias_pu200 = np.array(minbias_pu200file)
    
    if args.pileup == 0:
        PU0_Private = open("DsTau3muPU0_Private.pkl", "rb")
        tau_pu0file = pickle.load(PU0_Private)
        taudata = np.array(tau_pu0file)
        
        del tau_pu0file
        PU0_Private.close()

    elif args.pileup==200:
        PU200_MTD = open("DsTau3muPU200_MTD.pkl", "rb")
        tau_pu200file = pickle.load(PU200_MTD)
        tau_pu200 = np.array(tau_pu200file)

        del tau_pu200file
        PU200_MTD.close()
        taudata = filterPU200(tau_pu200)
        del tau_pu200
    
    maxhits = args.maxhits
    taudata = filterandpad(taudata, maxhits)
    bgdata = minbias_filterandpad(minbias_pu200,maxhits)
    
    trainModel(taudata, bgdata, maxhits, device, args)
    
if __name__ == '__main__':
    main()