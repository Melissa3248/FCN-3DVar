import torch
from tqdm import tqdm
import os
import sys
import inspect
sys.path.append(os.path.dirname(os.getcwd())) # Fix Python path

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

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
        parser.add_argument("--yaml_config", default=os.path.dirname(os.path.dirname(os.getcwd())) + '/FourCastNet/config/AFNO-ERA5-subset.yaml', type=str)
        parser.add_argument("--config", default='full_field', type=str)

        parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')

        # added these arguments (useful for FCN)
        parser.add_argument("--img_shape_x", default = 720, type=int)
        parser.add_argument("--img_shape_y", default = 1440, type=int)
        parser.add_argument("--n_features", default = 20, type=int)

        # added these arguments (useful for data assimilation)
        parser.add_argument("--analysis", type = str, help = "analysis state data filepath") # shape (1,N_ensem,20,720,1440)
        parser.add_argument("--y_obs_upscale", type = str, help = "observation data filepath") # shape (1,N_ensem,20,720/factor,1440/factor)
        parser.add_argument("--clean_init", type = str, help = 'path to ground truth data')
        parser.add_argument("--horizon", type = int)
        parser.add_argument("--save_dir" , type = str)
        parser.add_argument("--factor", type = int)
        
        args = parser.parse_args()
        params = YParams(os.path.abspath(args.yaml_config), args.config)

        # add image size parameters
        params['img_shape_x'] = args.img_shape_x
        params['img_shape_y'] = args.img_shape_y
        params['n_features'] = args.n_features


        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

        params['best_checkpoint_path'] = args.weights 

        n_ics = params['n_initial_conditions']
        
        float_type = torch.float32

        ########################################################################################################
        #                             Define FCN model, load validation data                                   #
        ########################################################################################################
        n_features = args.n_features
        lon = args.img_shape_y
        lat = args.img_shape_x

        FCN = setup(params)

        # read in mean and standard deviations
        m = torch.from_numpy(np.load(params.global_means_path))[:,:20,:,:]
        s = torch.from_numpy(np.load(params.global_stds_path))[:,:20,:,:]
        
        clean_init = h5py.File(args.clean_init)['fields'][567,:20,:720,:]
        clean_init = (torch.from_numpy(np.array(clean_init)) - m ) / s
        

        analysis = torch.from_numpy(np.load(args.analysis.format(args.factor))).to(device) # shape: (1,N_ensem, n_features, lat, lon )
        
        N_ensem = 50
        
        if not os.path.isdir(args.save_dir.format(args.factor)):
            os.makedirs(args.save_dir.format(args.factor))
        
        
        smoothed_obs = torch.from_numpy(np.load(args.y_obs_upscale.format(args.factor))).unsqueeze(0).expand((1,N_ensem,n_features,lat,lon)).to(device)
        smoothed_obs = smoothed_obs + torch.normal(0,0.3, size = smoothed_obs.shape).to(device)
        
        analysis = analysis.mean(1,keepdim = True).expand((1,N_ensem, n_features, lat, lon))
        analysis = analysis + torch.normal(0, 0.3, size = analysis.shape).to(device)
        
        clean_init = clean_init.unsqueeze(1).expand((1,N_ensem,n_features,lat,lon)).to(dtype=torch.float32).to(device)
        clean_init = clean_init + torch.normal(0,0.3, size = clean_init.shape).to(device)
        
        X_obs = smoothed_obs  
        X_analysis = analysis 
        X_clean = clean_init  
        
        # save the initialization
        np.save("{}/obs_preds{}.npy".format(args.save_dir.format(args.factor),0),  X_obs.squeeze().detach().cpu().numpy())
        np.save("{}/analysis_preds{}.npy".format(args.save_dir.format(args.factor),0),  X_analysis.squeeze().detach().cpu().numpy())
        np.save("{}/clean_preds{}.npy".format(args.save_dir.format(args.factor), 0), X_clean.squeeze().detach().cpu().numpy()) 
        
        for h in range(1,args.horizon+1):
            print(h)
            for n in range(N_ensem):
                X_obs[:,n,:,:,:] = FCN(X_obs[:,n,:,:,:])
                X_analysis[:,n,:,:,:] = FCN(X_analysis[:,n,:,:,:])
                X_clean[:,n,:,:,:] = FCN(X_clean[:,n,:,:,:]) 
                
            
            np.save("{}/obs_preds{}.npy".format(args.save_dir.format(args.factor),h),  X_obs.squeeze().detach().cpu().numpy())
            np.save("{}/analysis_preds{}.npy".format(args.save_dir.format(args.factor),h),  X_analysis.squeeze().detach().cpu().numpy())
            np.save("{}/clean_preds{}.npy".format(args.save_dir.format(args.factor), h), X_clean.squeeze().detach().cpu().numpy())
            