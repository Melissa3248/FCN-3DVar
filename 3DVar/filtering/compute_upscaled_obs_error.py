import h5py
import numpy as np
import torch
import argparse

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


parser = argparse.ArgumentParser()
parser.add_argument("--factor", type=int)
parser.add_argument("--folder", type = str)
args = parser.parse_args()

factor = args.factor

folder = args.folder.format(factor)


acc_obs = []
rmse_obs = []
factor = args.factor
for t in range(1459):
    try:
        if t % 50 == 0:
            print(t)
        upscaled_obs = np.load(folder + '/obs_upscaled{}.npy'.format(t))
        truth = h5py.File("../../data/2023.h5")['fields'][t+1,:,:,:]
        m = np.load("../../stats_v0/global_means.npy")[0,:20,:,:]
        s = np.load("../../stats_v0/global_stds.npy")[0,:20,:,:]
        truth = (truth-m)/s

        upscaled_obs = torch.from_numpy(upscaled_obs)
        truth = torch.from_numpy(truth).unsqueeze(0)
        
        acc_obs.append(weighted_acc_channels(upscaled_obs, truth))
        rmse_obs.append(weighted_rmse_channels(upscaled_obs, truth))
    except:
        break
    
np.save(folder + "/rmse_upscaled_obs.npy", np.array(rmse_obs))
np.save(folder + "/acc_upscaled_obs.npy", np.array(acc_obs))