'''
python upscale_obs.py --factor 8 --n_perturbations 20 --folder '/pscratch/sd/m/maadrian/output/ERA5/2023/vary_obs/new_framework_GLOBAL_FCNinterpyHnotIdyLdx_q0.5r0.0001n{}woodbury_qfactor{}GLOBALsmooth4_030324'
'''

import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--factor", type=int)
parser.add_argument("--folder", type = str)
parser.add_argument("--n_perturbations", type = int)
args = parser.parse_args()


def g_kernel(x, mean, sigma):
    return torch.exp(-(x-mean)@(x-mean)/(2*sigma**2))

factor = args.factor
n = args.n_perturbations
folder = args.folder

device = "cuda:0"


####################################
# define smoothing convolution weights, store them in params
h_size = factor 
half_hsize = int(h_size/2)
h = np.empty((2, h_size,h_size))
for i in range(h_size):
    for j in range(h_size):
        h[:,i,j] = (i,j)
mean = torch.ones(2,dtype = torch.float)*half_hsize
sigma = factor
gh = torch.empty(h.shape[1:])
for i in range(h.shape[1]):
    for j in range(h.shape[2]):
        gh[i,j]= g_kernel(torch.from_numpy(h[:,i,j]).to(dtype=torch.float), mean, sigma)
gh = gh/torch.sum(gh)
W_smooth = gh.unsqueeze(0).unsqueeze(0) 

chan = 1
n_features = 20
H_t = torch.nn.ConvTranspose2d(chan, chan, kernel_size = (factor,factor), stride = factor).to(device)
W =torch.ones((chan,chan,factor,factor)).to(device)
H_t.weight = torch.nn.Parameter( W.to(device), requires_grad = False)
H_t.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)

k_size = factor
smooth_conv = torch.nn.Conv2d(chan, chan, kernel_size=k_size, stride=1, padding='same', padding_mode = "replicate").to(device)
smooth_conv.weight = torch.nn.Parameter(W_smooth.to(device), requires_grad=False)
smooth_conv.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)

import os; print(os.getcwd())
for t in range(1459):
    print(n,t)

    obs = torch.from_numpy(np.load(folder.format(n,factor) + '/obs_perturb{}.npy'.format(t))).to(device)
    N_ensem = n
    n_features = 20
    lat = 720
    lon = 1440
    obs_large = torch.stack([H_t(obs[0,:,f,:,:].mean(0).unsqueeze(0)).squeeze() for f in range(20)]).unsqueeze(0).to(device)

    # smooth the observations
    smoothed_obs = torch.empty((n_features,lat,lon)).to(device)
    for f in range(n_features):
        smoothed_obs[f,:,:] = smooth_conv(obs_large[0,f,:,:].unsqueeze(0).unsqueeze(0).to(device))#.squeeze().to(device)
    np.save(folder.format(n,factor) + "/obs_upscaled{}.npy".format(t), smoothed_obs.unsqueeze(0).detach().cpu().numpy())


