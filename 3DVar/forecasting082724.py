'''
Visualizing the timepoints where forecasting failed while using a FCN-3DVar analysis as an initial condition

python forecasting082724.py --weights '/pscratch/sd/m/maadrian/output/ERA5/weights/backbone.ckpt' --horizon 20
'''
import torch
from tqdm import tqdm
import os
import sys
import inspect


sys.path.append(os.path.dirname(os.getcwd()))


from utils_fcn.YParams import YParams
from utils_fcn.data_loader_multifiles import get_data_loader
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

logging_utils.config_logger()


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

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

# standardize input X with means m and standard deviations s
def standardize(X, m, s):
    return (X-m)/s

# reverse standardize input X with means m and standard deviations s
def inv_standardize(Z,m,s):
    return Z*s+m

def forecast_to_horizon(X,model,horizon):
    '''
    X: initial condition, shape: (1,N_ensem, n_features, lat, lon)
    model: forecasting model
    horizon: number of timepoints to forecast into the future
    '''
    N_ensem = X.shape[1]
    

    for h in range(1,horizon+1):
        for n in range(N_ensem):
            X[:,n,:,:,:] = model(X[:,n,:,:,:])
            
    return X.mean(1) 
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
        parser.add_argument("--run_num", default='00', type=str)
        parser.add_argument("--yaml_config", default=os.path.dirname(os.getcwd()) + '/config/AFNO-ERA5-GLOBAL.yaml', type=str)
        parser.add_argument("--config", default='perturbations', type=str)

        parser.add_argument("--interp", default=0, type=float)
        parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')

        # added these arguments (useful for FCN)
        parser.add_argument("--img_shape_x", default = 720, type=int)
        parser.add_argument("--img_shape_y", default = 1440, type=int)
        parser.add_argument("--n_features", default = 20, type=int)
        parser.add_argument("--horizon", type = int)
    


        args = parser.parse_args()
        params = YParams(os.path.abspath(args.yaml_config), args.config)

        # add image size parameters
        params['img_shape_x'] = args.img_shape_x
        params['img_shape_y'] = args.img_shape_y
        params['n_features'] = args.n_features
        params['log_to_wandb'] = False



        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

        n_ics = params['n_initial_conditions']
        params['best_checkpoint_path'] = args.weights
        
        float_type = torch.float32

        ########################################################################################################
        #                             Define FCN model, load validation data                                   #
        ########################################################################################################
        n_features = args.n_features
        lon = args.img_shape_y
        lat = args.img_shape_x

        FCN = setup(params)
        N_ensem = 1 
        
        # read in mean and standard deviations
        m = torch.from_numpy(np.load(params.global_means_path))[:,:20,:,:]
        s = torch.from_numpy(np.load(params.global_stds_path))[:,:20,:,:]

        tpts = [849,850]
        
        preds = np.empty((len(tpts), args.horizon, n_features, lat, lon))
        
        for t in range(len(tpts)):
            # load init
            X = torch.from_numpy(np.load(f"/pscratch/sd/m/maadrian/output/ERA5/2023/vary_obs/new_framework_GLOBAL_FCNinterpyHnotIdyLdx_q0.5r0.0001n1woodbury_qfactor18GLOBALsmooth4_020425_minus4/update{tpts[t]}.npy")).to(device)

            for h in range(args.horizon):
                X = FCN(X.squeeze().unsqueeze(0))
                preds[t,h,:,:,:] = X.detach().cpu().numpy()
                
        import h5py 
        d = h5py.File('/pscratch/sd/m/maadrian/output/ERA5/2023/failure_preds_020425_minus4.h5', 'w')
        d['fields'] = preds
        d.close()
     