import torch
import numpy as np

def g_kernel(x, mean, sigma):
    return torch.exp(-(x-mean)@(x-mean)/(2*sigma**2))

def add_smoothing_conv_params(params):
    ####################################
    # define smoothing convolution weights, store them in params
    h_size = params.smooth_size
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
    params['smoothing_W'] = gh.unsqueeze(0).unsqueeze(0)
    ####################################
    return params
    
    
def smooth_convolution(params,device):
    chan = 1
    k_size = params.smoothing_W.shape[3]
    smooth_conv = torch.nn.Conv2d(chan, chan, kernel_size=k_size, stride=1, padding=int(k_size/2), padding_mode = "replicate").to(device)

    smooth_conv.weight = torch.nn.Parameter(params.smoothing_W.to(device), requires_grad=False)
    smooth_conv.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)
    return smooth_conv

def apply_smooth_conv(X, params,device):

    bs = X.shape[0]
    N_ensem = X.shape[1]
    n_features = X.shape[2]
    lat = X.shape[3]
    lon = X.shape[4]
    
    smooth_conv = smooth_convolution(params,device)
    
    X = torch.stack([smooth_conv(X[b,n,f,:,:].clone().unsqueeze(0).unsqueeze(0)).squeeze()[:lat,:lon] for b in range(bs) for n in range(N_ensem) for f in range(n_features)]).reshape(bs,N_ensem, n_features,lat,lon)
    return X 



def H_func(params, device):
    chan = 1 #params.n_features 

    W = torch.ones((chan,chan,params.patch_size,params.patch_size))/params.patch_size/params.patch_size

    H = torch.nn.Conv2d(chan, chan, kernel_size=params.patch_size, stride=params.patch_size).to(device)
    H.weight = torch.nn.Parameter(W.to(device), requires_grad=False)
    H.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)
    return H


def H_transpose(params, device):
    chan = 1
    H_t = torch.nn.ConvTranspose2d(chan, chan, kernel_size=params.patch_size, stride=params.patch_size).to(device)
    W = torch.ones((chan,chan,params.patch_size,params.patch_size))
    H_t.weight = torch.nn.Parameter(W.to(device), requires_grad=False)
    H_t.bias = torch.nn.Parameter(torch.zeros(chan).to(device), requires_grad = False)
    return H_t

def apply_H(X, params, device):
    half_patch = int(params.patch_size/2)
    repl_tl = torch.nn.ReplicationPad2d((half_patch,0,half_patch,0)).to(device)
    
    HX= torch.stack([H(repl_tl(X[b,n,f,:,:].clone().unsqueeze(0))[:,:params.img_shape_x, :params.img_shape_y]) for b in range(*bs)  for n in range(N_ensem) for f in range(n_features) ]).reshape(*bs, N_ensem, n_features*int(lat/8)*int(lon/8))

    return HX
    
    
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result