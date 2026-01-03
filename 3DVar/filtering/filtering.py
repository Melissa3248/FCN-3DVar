import numpy as np
import argparse
import h5py
import torch


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


###############################################################################################
#                             Set up ArgumentParser for FCN                                   #
###############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--path", type =str)
parser.add_argument("--data_path", type = str)
parser.add_argument("--max_t", type = int)
parser.add_argument("--file_tag", type = str)
parser.add_argument('--upscale', type = bool, default = False)
parser.add_argument("--factor", type = int)
parser.add_argument("--obs_error", type = bool, default = False)
args = parser.parse_args()

# read in original data
m=np.load('../../stats_v0/global_means.npy')[:,:20,:,:]
s=np.load('../../stats_v0/global_stds.npy')[:,:20,:,:]

factor = args.factor
#if not(upscale):
w_rmse_list = []
w_rmse_se_list = []
w_acc_list = []
w_acc_se_list = []

device = 'cuda:0'

for t in range(args.max_t):
    # read in original FCN analysis update
    try:
        X_train =  h5py.File('{}'.format(args.data_path.format(args.factor)))['fields'][t+1,:20,:720,:1440]
        X_train = (X_train-m[0,:,:,:])/s[0,:,:,:]
        pred = np.load("{}/update{}.npy".format(args.path.format(args.factor),t))

        # weighted RMSE
        if args.obs_error:
            w_rmse = weighted_rmse_channels(torch.from_numpy(np.array(pred[0,:,:,int(factor/2)::factor,int(factor/2)::factor]).to(device)), torch.from_numpy(X_train[:,int(factor/2)::factor,int(factor/2)::factor]).unsqueeze(0).to(device) ) # shape (n_ensem, channels)
        else: 
            w_rmse = weighted_rmse_channels(torch.from_numpy(np.array(pred[0,:,:,:,:])).to(device), torch.from_numpy(X_train).unsqueeze(0).to(device) ) # shape (n_ensem, channels)
        w_rmse_error = w_rmse.mean()
        w_rmse_se = 2*torch.std(w_rmse.mean(1))/np.sqrt(w_rmse.shape[0]) # std over ensemble dimension
        w_rmse_list.append(w_rmse_error.detach().cpu().numpy())
        w_rmse_se_list.append(w_rmse_se.detach().cpu().numpy())

        # weighted ACC
        if args.obs_error:
            w_acc = weighted_acc_channels(torch.from_numpy(np.array(pred[0,:,:,int(factor/2)::factor,int(factor/2)::factor])).to(device), torch.from_numpy(X_train[:,int(factor/2)::factor,int(factor/2)::factor]).unsqueeze(0).to(device) )
        else:
            w_acc = weighted_acc_channels(torch.from_numpy(np.array(pred[0,:,:,:,:])).to(device), torch.from_numpy(X_train).unsqueeze(0).to(device) )
        w_acc_error = w_acc.mean()
        w_acc_se =  2*torch.std(w_acc.mean(1))/np.sqrt(w_acc.shape[0]) # std over ensemble dimension
        w_acc_list.append(w_acc_error.detach().cpu().numpy())
        w_acc_se_list.append(w_acc_se.detach().cpu().numpy())

        print(t, w_acc.mean(), w_rmse.mean())
        
    except:
        break


# save the results
np.save(args.path.format(args.factor) + "/enkf_filtering_rmse_{}.npy".format( args.file_tag.format(args.factor)), np.array(w_rmse_list))
np.save(args.path.format(args.factor) + "/enkf_se_filtering_rmse_{}.npy".format( args.file_tag.format(args.factor)), np.array(w_rmse_se_list))

np.save(args.path.format(args.factor) + "/enkf_filtering_acc_{}.npy".format( args.file_tag.format(args.factor)), np.array(w_acc_list))
np.save(args.path.format(args.factor) + "/enkf_se_filtering_acc_{}.npy".format( args.file_tag.format(args.factor)), np.array(w_acc_se_list))


    