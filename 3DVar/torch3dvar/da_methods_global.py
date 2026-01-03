import torch
import math
from torch3dvar import misc
from torch.nn.functional import normalize
import sys
import os
import logging
import numpy as np
import psutil

# append path 2 folders above current folder
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))) # Fix Python path

from tqdm import tqdm
import time

def analysis_update(HX, obs_perturb, bs, N_ensem, X_ct, r,q, float_type, device, params, H, H_t):

    lat = params.img_shape_x
    lon = params.img_shape_y

    n_features = params.n_features

    factor = params.factor
    y_dim = n_features*int(lat/factor)*int(lon/factor)
    HBB_TH_T = params.HBB_TH_T.repeat(y_dim).to(device)
    ####################################################################

    inflation = 1/params.HBB_TH_T
    c = (inflation*q*HBB_TH_T + torch.tensor(r).repeat(y_dim).to(device))

    obs_perturb = obs_perturb.reshape((bs,N_ensem, n_features, int(lat/factor), int(lon/factor))).to(device)
    HX = HX.reshape((bs,N_ensem, n_features, int(lat/factor), int(lon/factor))).to(device)
    
    c = c.reshape((n_features, int(lat/factor), int(lon/factor))).to(device)
    
    B = params['B']
    H_t = params['H_t']

    part_obs = torch.stack([inflation*q*B(B(H_t((1/c[f,:,:]*obs_perturb[b,n,f,:,:]).unsqueeze(0).unsqueeze(0)))).squeeze() for b in range(bs) for n in range(N_ensem) for f in range(n_features)]).reshape(bs,N_ensem,n_features*lat*lon)
    
    part_x = torch.stack([-inflation*q*B(B(H_t((1/c[f,:,:]*HX[b,n,f,:,:]).unsqueeze(0).unsqueeze(0)))).squeeze() for b in range(bs) for n in range(N_ensem) for f in range(n_features)]).reshape(bs,N_ensem,n_features*lat*lon)

    return part_obs, part_x

def _3DVar(func, 
         obs_func, y_obs, N_ensem, init_m, init_C_param, model_Q_param, noise_R_param, device, params, 
                    init_X=None, 
                    save_filter_step={'mean'},
                    smooth_lag=0, t0=0., var_inflation=None, localization_radius=None, compute_likelihood=True, linear_obs=True, time_varying_obs=False,
                    save_first=False, tqdm=None, float_type = torch.float32, save_directory = None, H = None, H_t = None, **ode_kwargs):
    """
    3DVar

    Key args:
        ode_func (torch.nn.Module): Vector field f(t,x)
                Tip: Wrap all parameters of interest that you want to evaluate gradient by torch.nn.Parameter()
                NOTE: This implicitly assume the underlying latent model is an ODE. For generic type of latent evolutions x_{t+1}=F(x_t), slight modifications of the forcast step are required.
        obs_func (torch.nn.Module): Observation model h(x), assumed to be linear h(x) = Hx.
                If time varying_obs==True, can take a list of torch.nn.Module's
        t_obs (tensor): 1D-Tensor of shape (n_obs,). Time points where observations are available.
                This does NOT need to be time-uniform. By default, t0 is NOT included. Must be monotonic increasing.
        y_obs (tensor): Tensor of shape (n_obs, *bs, y_dim). Observed values at t_obs.
                '*bs' can be arbitrary batch dimension (or empty).
                Observations are assumed to have the same dimension 'y_dim'. However, observation model can be time-varying.
        N_ensem: Number of particles.
        init_m (tensor): Tensor of shape (x_dim, ). Mean of the initial distribution.
        init_C (noise.AddGaussian):  covariance of initial distribution
        model_Q_param (noise.AddGaussian): model error covariance
        noise_R_param (noise.AddGaussian): observation error covariance

    Optional args:
        init_X (tensor): Tensor of shape (*bs, N_ensem, x_dim). Initial ensemble if pre-specified.
        ode_method: Numerical scheme for forward equation. We use 'euler' or 'rk4'. Other solvers are available. See https://github.com/rtqichen/torchdiffeq
        ode_options: Set it to dict(step_size=...) for fixed step solvers for the forward equation. Adaptive solvers are also available - see the link above.
        adjoint (bool): Whether to compute gradient via adjoint equation or direct backpropagation through the solver.
        adjoint_method: Numerical scheme for adjoint equation if adjoint==True.
        adjoint options: Set it to dict(step_size=...) for fixed step solvers for the adjoint equation. Adaptive solvers are also available - see the link above.
        ode_kwargs: additional kwargs for neuralODE.
        save_filter_step:
            If contains 'mean', then particle means will be saved.
            If contains 'particles', then all particles will be saved.
            (Note: the up-to-date/final particles will always be returned seperately)
        t0: The timestamp at which the ensemble is initialized.
            By default, we DO NOT assume observation is available at t0. Slight modifications of the code are needed to handle this situation.
        var_inflation: See discussion in paper. Typical value is between 1 and 1.1. None by default.
        localization_radius: See discussion in paper. Typical value is 5. None by default.
        compute_likelihood: Whether to compute data log-likelihood in the filtering process.
                            Must be set to True for AD-EnKF.
        linear_obs: If set to True, then obs_func must be 'nn_templates.Linear' class. The observation model is y = Hx + noise where H is a matrix.
                    If set to False, then obs_func can be any differentiable function/module in PyTorch. The observation model is y = obs_func(x) + noise
        time_varying_obs: If set to False, the observation model is time-invariant. A single nn.Module/function is sufficient for the obs_func argument.
                        If set to True, the observation model can be different across time. A list of nn.Module/functions is needed for obs_func argument and has the same length as t_obs.
        save_first: Set it to True to save the initial ensemble.
        tqdm: Set tqdm=tqdm to use the tqdm format for presenting.

    Returns:
        X (tensor): Tensor of shape (*bs, N_ensem, x_dim). Final ensemble.
        res (dict):
            If save_filter_step contains 'mean', then res['mean'] will be tensor of shape (n_obs, *bs, x_dim)
            If save_filter_step contains 'particles', then res['particles'] will be tensor of shape (n_obs, *bs, N_ensem, x_dim)
        log_likelihood (tensor): Log likelihood estimate # (*bs)
    """

    x_dim = init_m.shape[0]
    y_dim = y_obs.shape[-1]
    n_obs = y_obs.shape[0]
    bs = y_obs.shape[1:-1]

    lat = params.img_shape_x
    lon = params.img_shape_y
    n_features = params.n_features
    
    zeta = params.zeta
    r = params.r
    q = model_Q_param.post_process(model_Q_param.q, model_Q_param.param_type)
    
    log_likelihood = torch.zeros(*bs, device=device) if compute_likelihood else None  # (*bs),  tensor(0.) if no batch dimension

    if init_X is not None:
        X = init_X.detach()
    else:
        X = init_C_param(init_m.expand(*bs, N_ensem, x_dim))


    res = {}
    if 'particles' in save_filter_step:
        res['particles'] = []
        res['particles'] = torch.empty(n_obs + 1, *bs, N_ensem, x_dim, dtype=init_m.dtype)
        res['particles'][0] = X.detach()
    if 'mean' in save_filter_step:
        X_m = X.mean(dim=-2)
        res['mean'] = torch.empty(n_obs + 1, *bs, x_dim, dtype=init_m.dtype)
        res['mean'][0] = X_m.detach()


    
    if not linear_obs:
        raise Exception("Not implemented")
        
    # read in max_inf_norms
    max_norms = torch.from_numpy(np.load("../max_inf_norms.npy")).to(device)
        

    pbar = tqdm(range(n_obs), desc="Running 3DVar", leave=False) if tqdm is not None else range(n_obs)
    for j in pbar:
        
        ################### Forecast step - FCN prediction#######################
        #switch from d_x to (*bs, N, features, lat, long) format
        
        X = X.reshape(*bs, N_ensem, n_features, lat, lon)
        X = torch.stack([func(X[i,n,:,:,:].clone().unsqueeze(0).to(dtype=float_type)) for i in range(*bs) for n in range(N_ensem)]).reshape(*bs, N_ensem, n_features,lat,lon)
            
            
        ############################ get ensemble dim back to the original size if needed
        # trim ensemble if one or more members are exploding
        trimmed_X = []
        for n in range(N_ensem):
            exceeds_norm_bool = torch.any(torch.stack([torch.any(X[i,n,f,:,:].reshape(lat*lon) > max_norms[f]) for i in range(*bs) for f in range(n_features)]))
            
            if not(exceeds_norm_bool):
                trimmed_X.append(X[:,n,:,:,:])
            if exceeds_norm_bool:
                logging.info("WARNING: FILTER DIVERGENCE")
                sys.exit()
        
        chan = 1
        
        k_size = params.smoothing_W.shape[3]
        smooth_conv = torch.nn.Conv2d(chan, chan, kernel_size=k_size, stride=1, padding=int(k_size/2), padding_mode = "replicate").to(device) # used to be replicate
        smooth_conv.weight = torch.nn.Parameter(params.smoothing_W.to(device), requires_grad=False)
        smooth_conv.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)
        
        params['smooth_conv'] = smooth_conv

        ######################################################################################################
        #### regularizer for X (aka smoothing over spatial domain)
        X = X.reshape(*bs, N_ensem, n_features, lat, lon)
        X = torch.stack([smooth_conv(X[b,n,f,:,:].clone().unsqueeze(0).unsqueeze(0)).squeeze()[:lat,:lon] for b in range(*bs) for n in range(N_ensem) for f in range(n_features)]).reshape(*bs,N_ensem, n_features*lat*lon)

        X = X.flatten(2)
        
        X_m = X.mean(dim=-2).unsqueeze(-2)  # (*bs, 1, x_dim)
        X_ct = X - X_m

        ################ Analysis step ##################
        y_obs_j = y_obs[j].unsqueeze(-2)  # (*bs, 1, y_dim)
        
        obs_perturb = y_obs_j.expand(*bs, N_ensem, y_dim) # noisy observation


        if linear_obs:
            X = X.reshape(*bs, N_ensem, n_features, lat, lon)
            factor = params.factor
            HX= X[:,:,:,int(factor/2)::factor,int(factor/2)::factor] 
            HX = HX.flatten(2)

            X = X.flatten(2)
            HX_m = HX.mean(1, keepdim =True).to(device)
            
            # compute the analysis update
            update_obs, update_x = analysis_update(HX, obs_perturb, *bs, N_ensem, X_ct, r,q,float_type, device, params, H,H_t)
            X = X + (update_obs + update_x)
            
            
            ######################################################################################################
            
            if save_directory is not None:
                np.save(save_directory + '/obs_perturb{}.npy'.format(int(t0+j)),
                        obs_perturb.reshape(1,N_ensem, n_features,int(lat/factor),int(lon/factor)).detach().cpu().numpy())
                
                np.save(save_directory + "/update{}.npy".format(int(t0+ j )),
                        X.reshape(1,params.n_perturbations, n_features, lat, lon).detach().cpu().numpy())
    
            # compute log likelihood
            if compute_likelihood:
                raise Exception("Not implemented")
            
        else:
            raise Exception("Not implemented")
            

        if 'particles' in save_filter_step:
            res['particles'][j+1] = X.detach()
        if 'mean' in save_filter_step:
            X_m = X.mean(dim=-2)
            res['mean'][j+1] = X_m.detach()

    if not save_first:
        for key in res.keys():
            res[key] = res[key][1:]
    return X, res, log_likelihood
