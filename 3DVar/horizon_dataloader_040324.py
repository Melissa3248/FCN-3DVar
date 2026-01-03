
import torch
from tqdm import tqdm
import os
import sys
import inspect

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from utils_fcn.YParams import YParams
from networks.afnonet import AFNONet
from utils_fcn import logging_utils
import argparse
import torch.distributed as dist
import logging
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import glob
import h5py
import psutil

from da_utils import weighted_rmse_channels, weighted_acc_channels 
from torch.utils.data import DataLoader


logging_utils.config_logger()


def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    if params["orography"]:
        params['N_in_channels'] = n_in_channels + 1
    else:
        params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[0, out_channels] # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]
    
    
    # load the model
    if params.nettype == 'afno':
        model = AFNONet(params).to(device)
    else:
        raise Exception("not implemented")
    
    checkpoint_file  = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)
    return model 


from torch.utils.data import Dataset

def find_between(s, start, end):
    return int((s.split(start))[1].split(end)[0])

class HorizonData(Dataset):
    def __init__(self, params, data_path, data_name, horizon, means, stds, mask, length=0):
        self.params = params
        self.data_path = data_path
        self.files_paths = glob.glob(data_path + "/*.npy")
        # filter out uneeded data
        self.files_paths = [self.files_paths[t] for t in range(len(self.files_paths)) if data_name in self.files_paths[t]]
        print(self.data_path)
        
        print(len(self.files_paths))
        if '{}_t'.format(data_name) in self.files_paths[0]:
            self.files_paths.sort(key=lambda x: find_between(x, data_path + '/{}_t'.format(data_name), '.npy'))
            self.files_paths = self.files_paths[length-1:] ############################# trim off original data that we dont have

        elif '{}'.format(data_name) in self.files_paths[0]:
            self.files_paths.sort(key=lambda x: find_between(x, data_path + '/{}'.format(data_name), '.npy'))
            self.files_paths = self.files_paths[length-1:] ############################# trim off original data that we dont have

        elif 'X0_' in self.files_paths[0]:
            self.files_paths.sort(key=lambda x: find_between(x, data_path + '/X0_', '.h5'))
        elif 'mask_t' in self.files_paths[0]:
            self.files_paths.sort(key=lambda x: find_between(x, data_path + '/mask_t', '.h5'))
            self.files_paths = self.files_paths[length-1:] ############################# trim off original data that we dont have 
                                                           ############################# corresponding initial conditions for

        else:
            self.files_paths.sort()
        
        logging.info("Getting file stats from {}".format(self.files_paths[0]))
        self.n_samples_per_file = np.load(self.files_paths[0]).shape[0]

        self.horizon = horizon
        self.n_features = params.n_features
        
        self.img_shape_x = params.img_shape_x
        self.img_shape_y = params.img_shape_y
        
        self.means = means
        self.stds = stds
        self.mask = mask
        
        
    def __len__(self):
        return self.n_samples_per_file * len(self.files_paths)
    
    def standardize(self, data):
        if len(data.shape) ==5: # batch size, horizon, n_features, lat, lon
            return (data-self.means.unsqueeze(0))/self.stds.unsqueeze(0)
        else:
            return (data-self.means)/self.stds
    
    def __getitem__(self, idx):
        '''
        idx: integer, specifies the index for the timepoint that we would like to pull idx to idx+horizon data from 
        Returns an object of shape (horizon, n_features, img_shape_x, img_shape_y)
        '''
        # get index of which file contains the timepoint idx
        # assumes all files in filepath have the same # of tpts per file
        file_loc = int(idx/self.n_samples_per_file)
        
        # keep track of how many timepoints we've stored so far
        count = 0
        
        # store horizon data as we load it
        data = [] 
        
        # iterate through the files until we've pulled all relevant data (up to the specified horizon)
        while count < self.horizon:
            # open the file for the current timepoint index
            d= np.load(self.files_paths[file_loc])

            
            # pull the timepoints that we need from the file
            added_data = d[file_loc- file_loc*self.n_samples_per_file:file_loc- file_loc*self.n_samples_per_file+self.horizon,:,:,:]

            # add the pulled data to our storage list
            data.append(torch.tensor(added_data))
            
            # add how many timepoints we've stored so far
            count += added_data.shape[0]
            
            # if we haven't reached the horizon with this dataset, store the new index to repeat
            # the process of pulling data
            idx = idx + added_data.shape[0] 
            
            # get a new file location for the next step
            file_loc = int(idx/self.n_samples_per_file)
            
            # check if we've exceeded the horizon with the # of timepoints we've pulled
            if count > self.horizon:
                # if we have, there's a bug in this code
                print("something's wrong")
                sys.exit("something's wrong")
        if self.mask:
            return torch.cat(data,dim=0) # don't need to normalize mask data since it's binary
        else:
            print("correct")
            return self.standardize(torch.cat(data, dim=0))
        
def get_data_loader(params, data_path, data_name, horizon,m,s, mask = False, length=  0):
    dataset = HorizonData(params, data_path, data_name, horizon,m, s, mask, length)
    dataloader= DataLoader(dataset, batch_size = int(params.batch_size),
                           #num_workers=1,
                          num_workers=params.num_data_workers,
                          shuffle = False, sampler = None,
                          pin_memory =torch.cuda.is_available())
    return dataloader, dataset
        
    
if __name__ == "__main__":

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")


        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ###############################################################################################
        #                             Set up ArgumentParser for FCN                                   #
        ###############################################################################################
        parser = argparse.ArgumentParser()
        parser.add_argument("--yaml_config", type=str)
        parser.add_argument("--config", default='full_field', type=str)
        parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights')
        parser.add_argument("--tag", default = "", type = str)
        parser.add_argument('--save_folder', type=str)
        parser.add_argument('--init_filepath', type = str, default = None)

        args = parser.parse_args()
        params = YParams(os.path.abspath(args.yaml_config), args.config)


        params['weights'] = args.weights
        params['save_folder'] = args.save_folder
        params['init_filepath'] = args.init_filepath

        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        
        params['best_checkpoint_path'] = params.weights

        params.log()

        n_ics = params['n_initial_conditions']
        
        ################################################################################################
        FCN = setup(params)
        float_type = torch.float32
        n_features = params.n_features
        lat = params.img_shape_x
        lon = params.img_shape_y
        
        # read in mean and standard deviations
        m = torch.from_numpy(np.load(params.global_means_path))[:,:20,:,:]
        s = torch.from_numpy(np.load(params.global_stds_path))[:,:20,:,:]
        
        
        m_init = torch.zeros(20).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        s_init = torch.ones(20).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
        horizon = params.horizon # forecast horizon
    
        
        if params.clean_init:
            gt_length = 1
        elif params.data_name == "obs_upscaled":
            gt_length = 2
        else:
            gt_length = params.length+1
        
        gt_loader, gt_data = get_data_loader(params, "../../data/individual_files",data_name="2023", horizon =horizon,m=m,s=s, length = gt_length)
        
        if not params.clean_init:
            

            
            # load initial condition
            init_loader, init_data = get_data_loader(params, params.init_filepath,data_name=params.data_name, horizon =1,m=m_init if not(params.data_name == "climatology") else m ,s=s_init if not(params.data_name == "climatology") else s, length = params.length)
            init_iterator = iter(init_data)
                    
        forecast_errors = [] # (timepoints, prediction horizon, n_features)
        forecast_accs = []

        for i, sample in enumerate(gt_data): # i indexes the current batch, sample contains the out-of-sample data

            # sample has shape tpts, features, lat, lon
            logging.info("sample {}".format(i))
            tpts = sample.shape[0] - 1
            
            if params.clean_init:
                # add a time dimension and an ensemble dimension (ensemble dimension is 1 for a clean initialization)
                # X shape: (1, 1, n_features, lat, lon)
                X = sample[0,:,:,:].to(device,dtype = float_type).unsqueeze(0).unsqueeze(0) # initial condition for the FCN predictions
            else: 
                if i >= init_data.__len__(): # our filtering initial conditions may have less data than the 
                    break
                X = next(init_iterator).to(device,dtype=float_type) # set as initial condition
                if params.data_name in ["climatology", "obs_upscaled"]:
                    X = X.unsqueeze(1)

            horizon_errors = [] # should eventually be a matrix of shape (prediction horizon, n_features)
            horizon_accs = []

            for t in range(1,tpts+1): # we skip the first timepoint in the sample 

                pred = [FCN(X[:,n,:,:,:]) for n in range(X.shape[1])] # forward pass
                pred = torch.stack(pred, axis=1)
                
                horizon_errors.append(weighted_rmse_channels(pred.mean(1), sample[t,:,:,:].to(device,dtype = float_type).unsqueeze(0)).squeeze())
                horizon_accs.append(weighted_acc_channels(pred.mean(1), sample[t,:,:,:].to(device,dtype = float_type).unsqueeze(0)).squeeze())
                
                X = pred 
                
            horizon_errors = torch.stack(horizon_errors)
            horizon_accs = torch.stack(horizon_accs)
            forecast_errors.append(horizon_errors)
            forecast_accs.append(horizon_accs)
           
        forecast_errors = torch.stack(forecast_errors)
        forecast_accs = torch.stack(forecast_accs)

        if not os.path.exists(params.save_folder):
            os.mkdir(params.save_folder)
            
        if not os.path.exists(params.save_folder ):
            os.mkdir(params.save_folder )

        
        with h5py.File(params.save_folder  + "/horizon_mse_results_fcn-{}-{}.h5".format(params.data_name,args.tag),'w') as f:
            f['horizon_rmse'] = forecast_errors.detach().cpu().numpy()
            f['horizon_rmse_avg_by_features'] = torch.mean(forecast_errors,axis = -1).detach().cpu().numpy()
            f['horizon_acc'] = forecast_accs.detach().cpu().numpy()
            f['horizon_acc_avg_by_features'] = torch.mean(forecast_accs,axis=-1).detach().cpu().numpy()