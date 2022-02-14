import numpy as np
from utils import filterandpad
import math
from tqdm import tqdm
import os

def valid_filterandpad(dataframe, maxhits, variables, one_endcap=False, transform=False, naive=False):
    dataset = []
    
    if transform==True:
        if 'mu_hit_sim_phi' in variables:
            variables.remove('mu_hit_sim_phi')
        
        if naive == False:
            if 'mu_hit_x' not in variables and 'mu_hit_y' not in variables:
                variables.insert(0,'mu_hit_x')
                variables.insert(1,'mu_hit_y')
                
        if naive == True:
            if 'mu_hit_cos_phi' not in variables and 'mu_hit_sin_phi' not in variables:
                variables.insert(0,'mu_hit_cos_phi')
                variables.insert(1,'mu_hit_sin_phi')
        
    for i in tqdm(range(len(dataframe)), desc='Filtering Dataset'):
        
        event = dataframe.iloc[i]
        #print(event['n_gen_tau'])
        
        #print('i kept going in the loop')
        if int(event['n_mu_hit']) == 0:
            continue
        
        if transform==True:
            phiTransform(event, naive=naive)
            
        new_event = []
        
        unfiltered_event = [event[key] for key in variables]
        stations = event['mu_hit_station']
        is_neighbor = event['mu_hit_neighbor']
        
        for char in unfiltered_event:
            new_char = []
            for idx, hit in enumerate(char):
                if stations[idx] == 1 and is_neighbor[idx] == 0:
                    new_char.append(hit)
                
            while len(new_char) < maxhits:
                new_char.append(0)
                
            if len(new_char) > maxhits:
                new_char = new_char[0:maxhits]
            
            new_event.append(new_char)
        
        new_event = np.array(new_event)
        new_event = new_event.flatten()
        dataset.append(new_event)
        
    return dataset

def reorderEta(eta_vec):
    idx_dict = dict({})
    eta_vec = list(eta_vec)
    for i in range(len(eta_vec)):
        idx_dict[i] = eta_vec[i]
    
    eta_vec.sort()
    sorted_idxs = []
    
    for hit in eta_vec:
        for key in idx_dict:
            if idx_dict[key] == hit:
                sorted_idxs.append(key)
    
    return sorted_idxs

def reorderHits(row):
    interested_vars = ['mu_hit_sim_phi', 'mu_hit_sim_eta', 'mu_hit_sim_r', 'mu_hit_sim_theta', 'mu_hit_sim_z', 'mu_hit_bend', 'mu_hit_ring', 'mu_hit_quality', 'mu_hit_station', 'mu_hit_neighbor']
    eta = row['mu_hit_sim_eta']
    
    idxs = reorderEta(eta)
   
    for key in interested_vars:
        row[key] = row[key][idxs]
    
    return row
def genNpArrays(phiTransform=False, naive=False, comment=None):
    PU200_file = open("DsTau3muPU200_MTD.pkl", "rb")
    SignalPU200 = pickle.load(PU200_file)
    PU200_file.close()

    MinBias_file = open("MinBiasPU200_MTD.pkl", "rb")
    BgPU200 = pickle.load(MinBias_file)
    MinBias_file.close()
    
    interested_vars = ['mu_hit_sim_phi', 'mu_hit_bend']
    
    SignalPU200.apply(reorderHits, axis=1)
    BgPU200.apply(reorderHits, axis=1)
    
    SignalPU200['gen_mu_phi'] = SignalPU200['gen_mu_phi'] % (2*math.pi)
    SignalPU200['mu_hit_sim_phi'] = SignalPU200['mu_hit_sim_phi'].map(lambda x: np.radians(x%360)%(2*math.pi))
    SignalPU200['mu_hit_sim_theta'] = SignalPU200['mu_hit_sim_theta'].map(lambda x: np.radians(x%360)%(2*math.pi))
    
    BgPU200['mu_hit_sim_phi'] = BgPU200['mu_hit_sim_phi'].map(lambda x: np.radians(x%360)%(2*math.pi))
    BgPU200['mu_hit_sim_theta'] = BgPU200['mu_hit_sim_theta'].map(lambda x: np.radians(x%360)%(2*math.pi))
    
    train_signal = SignalPU200[0:int(.95*len(SignalPU200))]
    valid_signal= SignalPU200[int(.95*len(SignalPU200)):len(SignalPU200)]
    
    train_signal.sample(frac=1).reset_index(drop=True)
    valid_signal.sample(frac=1).reset_index(drop=True)
    
    train_signal_npy = filterandpad(train_signal, 50, interested_vars, transform=phiTransform, naive=naive)
    valid_signal_npy = valid_filterandpad(valid_signal, 50, interested_vars, transform=phiTransform, naive=naive)
    
    np.save('SignalPU200_'+comment+'_Train.npy', train_signal_npy)
    np.save('SignalPU200_'+comment+'_Valid.npy', valid_signal_npy)
    
    print("Signal Completed")
    
    train_bg = BgPU200[0:int(4.92*len(train_signal_npy))]
    valid_bg = BgPU200[int(4.92*len(train_signal_npy)):]
    
    train_bg.sample(frac=1).reset_index(drop=True)
    valid_bg.sample(frac=1).reset_index(drop=True)
    
    bg_train_npy = filterandpad(train_bg, 50, interested_vars, transform=phiTransform, naive=naive)
    bg_valid_npy = valid_filterandpad(valid_bg, 50, interested_vars, transform=phiTransform, naive=naive)
    
    current_dir = os.getcwd()
    
    
    np.save('BgPU200_'+comment+'_Train.npy', bg_train_npy)
    np.save('BgPU200_'+comment+'_Valid.npy', bg_valid_npy)
    
    print("Background Completed. Done")
    
genNpArrays()
