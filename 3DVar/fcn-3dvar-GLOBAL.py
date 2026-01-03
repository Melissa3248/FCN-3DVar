import torch
from tqdm import tqdm
import os
import sys
import inspect
import math
import torch.nn as nn
from time import process_time
from torch3dvar import nn_templates, noise
from torch3dvar import da_methods_global as da_methods

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


def g_kernel(x, mean, sigma):
    return torch.exp(-(x-mean)@(x-mean)/(2*sigma**2))

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
    #model.train()
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

        # added these arguments
        parser.add_argument("--img_shape_x", default = 720, type=int)
        parser.add_argument("--img_shape_y", default = 1440, type=int)
        parser.add_argument("--n_features", default = 20, type=int)
        parser.add_argument("--y_obs", type = str, help = "observation data filepath")
        parser.add_argument("--q", type = float)
        parser.add_argument("--zeta", type = float, default = 0, help = 'covariance inflation constant: (1 + zeta)*C_t')
        parser.add_argument("--factor", type = int, default = 1)
        parser.add_argument("--save_directory", type = str, default = None)
        parser.add_argument("--n_perturbations", type = int, default = 1)
        parser.add_argument("--X0_init", type = str)

        args = parser.parse_args()
        params = YParams(os.path.abspath(args.yaml_config), args.config)

        # add image size parameters
        params['img_shape_x'] = args.img_shape_x
        params['img_shape_y'] = args.img_shape_y
        params['n_features'] = args.n_features
        params['q'] = args.q
        params['zeta'] = args.zeta
        params['factor'] = args.factor
        params['save_directory'] = args.save_directory
        params['n_perturbations'] = args.n_perturbations
        params['X0_init'] = args.X0_init

        params['world_size'] = 1
        params['global_batch_size'] = params.batch_size
        
        lat = params['img_shape_x']
        factor = params['factor']
        lon = params['img_shape_y']
        

        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

        params['best_checkpoint_path'] = args.weights 
        params['resuming'] = False
        params['local_rank'] = 0

        n_ics = params['n_initial_conditions']

        ########################################################################################################
        #                             Define FCN model, load validation data                                   #
        ########################################################################################################
        learned_func = setup(params)

        train_size = 1
        N_ensem = params.n_perturbations # change number of ensemble members here or the config file

        # read in mean and standard deviations
        m = torch.from_numpy(np.load(params.global_means_path))[:,:20,:,:].to(device).to(dtype=torch.float)
        s = torch.from_numpy(np.load(params.global_stds_path))[:,:20,:,:].to(device).to(dtype=torch.float)
       

        ############################################################################################
        #                             Process observational data                                   #
        ############################################################################################
        
        float_type = torch.float32
        
        # read in y observation training data from h5 file
        y_obs_train = h5py.File(args.y_obs.format(args.factor), 'r') 

        # convert to torch tensor
        y_obs_train = torch.from_numpy(np.array(y_obs_train['fields'][:,:20,:,:])).to(device, dtype=float_type)
        
        logging.info("print y_obs_train shape: {}".format(y_obs_train.shape))
        
        # standardize y_obs_train
        y_obs_train  = (y_obs_train-m)/s

        # define number of observations in training set and the dimension of y
        n_obs = y_obs_train.shape[0]
        y_dim = np.prod(y_obs_train.shape[1:])

        # flatten the feature, latitude, and longitude dimensions
        y_obs_train = y_obs_train.flatten(1) # (n_obs, y_dim)

        # add train_size dimension
        y_obs_train = y_obs_train.unsqueeze(1)
        y_obs_train = y_obs_train.expand(n_obs,train_size, y_dim) # (n_obs, train_size, y_dim)
        

        logging.info("loaded y training data")
        logging.info("y shape: {}".format(y_obs_train.size()))

        ############################################################################################
        #                              Define initial condition X                                  #
        ############################################################################################

        # compute X dimension
        x_dim = params.img_shape_x*params.img_shape_y*params.n_features

        # define initial condition
        X0 = torch.from_numpy(np.load(params.X0_init.format(factor))).to(device)

        # standardize X
        # means and stds shape: (1, n_features, 1, 1)
        # X shape: (1, n_features, lat, long)
        X0 = (X0 - m)/s
        
        # flatten to shape (x_dim)
        X0 = X0.flatten()

        logging.info("loaded X training data")

        ############################################################################################
        #                                     Load H, Q, R, m, C                                   #
        ############################################################################################

        true_obs_func = None
        
        model_Q_true = noise.AddGaussian(x_dim, torch.sqrt(torch.tensor(params.q)), param_type='scalar').to(device)      
        params['r'] = 0.0001 
        noise_R_true = noise.AddGaussian(y_dim, torch.tensor(params.r), param_type='scalar').to(device)  # Gaussian perturbation 
        init_m = X0
        
         
        
        # since y_obs_train are clean obs, add noise to them
        y_obs_train = y_obs_train + torch.normal(0,np.sqrt(params.r), size = y_obs_train.shape).to(device)
        logging.info("WARNING: Added noise to the observations")
        
        
        ####################################
        # define smoothing convolution weights, store them in params
        h_size =  4 
        half_hsize = int(h_size/2)
        h = np.empty((2, h_size,h_size))
        for i in range(h_size):
            for j in range(h_size):
                h[:,i,j] = (i,j)
        mean = torch.ones(2,dtype = torch.float)*half_hsize
        sigma = 8
        gh = torch.empty(h.shape[1:])
        for i in range(h.shape[1]):
            for j in range(h.shape[2]):
                gh[i,j]= g_kernel(torch.from_numpy(h[:,i,j]).to(dtype=torch.float), mean, sigma)
        gh = gh/torch.sum(gh)
        params['smoothing_W'] = gh.unsqueeze(0).unsqueeze(0) #torch.ones(1,1,h_size,h_size)/h_size**2#
        ####################################
        
        # define smoothing convolution weights, store them in params
        h_size = factor
        half_hsize = int(h_size/2)
        h = np.empty((2, h_size,h_size))
        for i in range(h_size):
            for j in range(h_size):
                h[:,i,j] = (i,j)
        mean = torch.ones(2,dtype = torch.float)*half_hsize
        sigma = 8
        gh = torch.empty(h.shape[1:])
        for i in range(h.shape[1]):
            for j in range(h.shape[2]):
                gh[i,j]= g_kernel(torch.from_numpy(h[:,i,j]).to(dtype=torch.float), mean, sigma)
        gh = gh/torch.sum(gh)
        params['C_W'] = gh.unsqueeze(0).unsqueeze(0)
        
        init_C_param = None

        X = init_m.expand(train_size, N_ensem, x_dim).to(device, dtype = float_type)
                
        # make directory to save results
        d = params.save_directory + '_q{}r{}n{}factor{}GLOBAL'.format(params.q, params.r, N_ensem, params['factor'])
        if not os.path.exists(d):
            os.mkdir(d)
            
        chan = 1 

        kernel_size = (factor, factor)

        B = nn.Conv2d(chan, chan, kernel_size=kernel_size, stride=1, padding='same', padding_mode = 'replicate').to(device)
        B.weight = torch.nn.Parameter(params.C_W.to(device), requires_grad = False)
        B.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)
        
        H_t = nn.ConvTranspose2d(chan, chan, kernel_size = kernel_size, stride = kernel_size).to(device)
        W =torch.zeros((chan,chan,factor,factor)).to(device)
        W[:,:,int(factor/2),int(factor/2)]= 1
        H_t.weight = torch.nn.Parameter( W.to(device), requires_grad = False)
        H_t.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)

        H = nn.Conv2d(chan, chan, kernel_size=kernel_size, stride=kernel_size).to(device)
        W_H = torch.zeros(kernel_size).unsqueeze(0).unsqueeze(0).to(device)
        W_H[:,:,int(factor/2),int(factor/2)]=1

        H.weight = torch.nn.Parameter(W_H.to(device), requires_grad = False)
        H.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)
        
        
        params['B'] = B
        params['H_t'] = H_t
        params['H'] = H
        params['HBB_TH_T'] = torch.sum(params.C_W**2) 
        
        ############################################################################################
        #                                           3DVar                                           #
        ############################################################################################

        t_obs = range(n_obs)

        t0 = 0.
        L = 1 

        train_log_likelihood = torch.zeros(train_size, device=device)
        train_state_est_loss = torch.zeros(params.n_features,device = device) 
        t_start = t0

        logging.info("starting EnKF")


        for start in range(0, n_obs, L):
            end = min(start + L, n_obs)
            logging.info("({} to {})/{}".format(start,end,n_obs))
            t1_start = process_time() 

            # X: constructed initial condition (real initial condition for t=0), shape: (N_ensem, d_x)
            # res: reconstructed states for L time points, shape: (L, N_ensem, d_x)
            # log-likelihood: LL of the reconstruction given the data, shape: scalar
            X, res, _ = da_methods._3DVar(func = learned_func, obs_func = true_obs_func, t_obs = t_obs[start:end], 
                                                     y_obs = y_obs_train[start:end],N_ensem = N_ensem, 
                                                     init_m = init_m, init_C_param = init_C_param, model_Q_param = model_Q_true, 
                                                     noise_R_param = noise_R_true, device = device,  params =params, 
                                                     save_filter_step={'particles'}, t0=t_start, init_X=X,  tqdm=None,
                                                     compute_likelihood = False, float_type = float_type, 
                                                     save_directory = d)
            logging.info("X mean after enkf {}".format(X.mean()))
            t_start = t_start + L
            res['particles'] = res['particles'].reshape(end-start,N_ensem, params.n_features, params.img_shape_x, params.img_shape_y)
            logging.info("Time for processing 1 timepoint {}".format(process_time()-t1_start))

        logging.info("train state loss: {}".format(train_state_est_loss))

        print("END: X shape: {}".format(X.shape))
        #################################################################################################################
