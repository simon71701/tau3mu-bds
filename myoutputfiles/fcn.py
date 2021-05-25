import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        self.batch_size = 3*args.maxhits
        self.dropout = args.dropout
    
        self.hidden0 = nn.Sequential( 
            nn.Linear(self.batch_size, 200),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.out = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.out(x)
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
    parser.add_argument('--maxhits', type=int,
                        help="Maximum number of hits allowed")

    parser.set_defaults(batch_size=10,
                        dropout=.3,
                        epochs=150,
                        lr=.001,
                        pileup=0,
                        maxhits = 50)

    return parser.parse_args()

def filterPU200(dataset):
    singledOut = []
    
    for event in dataset:
        if event[0] == 1:
            singledOut.append(event)
    
    return singledOut

def collectChar(index, dataset):
    char = []
    
    for i in range(len(dataset)):
        char.append(dataset[i][index])
    
    
    return char

# Ordered (Phi, Eta, R)
def filterandpad(dataset, maxhits):
    filtered = []
    for event in dataset:
        chars = []
        if event[2] <= maxhits:
            for i in (29, 31, 32):
                char = np.pad(event[i], (0,maxhits-len(event[i])), 'constant', constant_values=0)
                chars.append(char)
            filtered.append(chars)
    return filtered

def minbias_filterandpad(minbias,maxhits):
    filtered = []
    for event in minbias:
        chars = []
        if event[0] <= maxhits:
            for i in (17, 19, 20):
                char = np.pad(event[i], (0,maxhits-len(event[i])), 'constant', constant_values=0)
                chars.append(char)
            filtered.append(chars)
    
    print(len(filtered))
    return filtered    

def normBatch(batch):
    normbatch = batch[0]/max(batch[0])
    for i in range(1, batch.size()[0]):
        torch.cat((normbatch, batch[i]/max(batch[i])))
        print(batch[i][0:10])
        print(normbatch[1][0:10])
        break
    return normbatch

def getMaxHits(dataset):
    maxhits = max(collectChar(2, dataset))
    
    return maxhits

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

def trainModel(dataset, bgdataset, args):
    classifier = Net(args)
    
    dataset = torch.tensor(dataset)
    bgdataset = torch.tensor(bgdataset)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    fake_data_loader = torch.utils.data.DataLoader(bgdataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    errors = []
    opt = torch.optim.Adam(classifier.parameters(),lr=.001)
    
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for batchnum, batch in enumerate(data_loader):
            error, real, fake = train_one_batch(classifier, opt, batch, fake_data_loader, args)
            
        print("Epoch {0} Finished: Total Error={1} ".format(epoch+1, error))
        errors.append(float(error))
                                         
    plt.plot(range(1,args.epochs+1), list(errors))
    plt.xlabel("Epoch")
    plt.title("Error Curve")
    plt.ylabel("Total Error")
    plt.savefig("ErrorCurve.png")

def main():
    
    args = arg_parse()
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
    
    maxhits = args.maxhits)
    taudata = filterandpad(taudata, maxhits)
    bgdata = minbias_filterandpad(minbias_pu200, maxhits)
    trainModel(taudata, bgdata, args)
    
if __name__ == '__main__':
    main()
