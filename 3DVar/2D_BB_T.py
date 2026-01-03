'''
Example usage of this file: 

python 2D_BB_T.py --lat 64 --lon 64 --factor_x 8 --factor_y 8 --tag "2D" --save_folder '/pscratch/sd/m/maadrian/Ens3DVar/example2D/'
'''

import torch
import argparse
import numpy as np
np.random.seed(10)
import logging

        
def g_kernel(x, mean, sigma):
    return torch.exp(-(x-mean)@(x-mean)/(2*sigma**2))


def create_BB_T(lat,lon,factor_x, factor_y,tag, save_folder):

    x = torch.empty((2,lat,lon))
    orig_x = x

    # create coordinates for the lats and lons in place
    for i in range(lat):
        for j in range(lon):
            x[0,i,j] = i
            x[1, i,j] = j

    x= x.unsqueeze(0)

    # pad x with half the patch size -- this is so we have the same number of patches as observations
    patch_size_x = factor_x
    patch_size_y = factor_y
    pad = torch.nn.ReplicationPad2d((int(patch_size_x/2),int(patch_size_y/2)-1,int(patch_size_x/2),int(patch_size_y/2)-1))
                                    #(padding_left, padding_right, padding_top, padding_bottom)
        
    x = pad(x)

    # create a sequence of factor by factor patches with the 2D coordinates
    kc, kh, kw = 2, factor_x, factor_y  # kernel size
    dc, dh, dw = 2, 1, 1  # stride
    patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw).squeeze()
    
    print(patches.shape)
    #import sys; sys.exit()

    flat_x = orig_x.flatten(1)

    # create a dictionary mapping the 2D coordinate to the vectorized index
    d = {}
    for i in range(flat_x.shape[1]):
        d["{}".format(flat_x[:,i].tolist())] = i
    logging.info("created dictionary")

    idx = [] # shape: [2, #patches*factor*factor]
    for i in range(patches.shape[0]):
        for j in range(factor_x):
            for k in range(factor_y):
                idx.append( [i, d["{}".format(patches[i,:,j,k].tolist())]]) # patches shape: [#patches, coords (2), factor, factor]
                # this lists out the image pixel indices involved for each patch, in order of patches

    
    logging.info("created indices")
    
    
    # create weights for the kernel -- based on truncated Gaussian
    # define smoothing convolution weights, store them in params
    h_size =  factor_x
    print("warning: assuming that factor_x = factor_y, change this later")
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
    W = gh 
    
    uniq_idx= torch.Tensor(idx) 
    values = W.flatten().repeat(int(uniq_idx.shape[0]/(factor_x*factor_y))) # values correspond to a truncated Gaussian kernel 
    

    # save sparse representation of B
    s = torch.sparse_coo_tensor(list(zip(*uniq_idx.tolist())), values.tolist(), (int(lat*lon), int(lat*lon) ))
    # duplicate indices in the uniq_idx list will add the corresponding values -- this is exactly what we want!!! (for the boundaries)

    logging.info("created sparse B matrix")    
    

    #randomly select locations to keep
    y_dim = int(lat/8*lon/8)
    keep = np.random.choice(range(lat*lon), size = y_dim, replace=False)
    H_bool = [True if i in keep else False for i in range(lat*lon)]
    
    np.save( save_folder + "/H_bool{}.npy".format(tag), np.array(H_bool))
    

    B = s.to("cuda:0")

    
    idxs = []
    for i in range(lat*lon):
        if H_bool[i]:
            idxs.append(i)
           
    # create new indices mapping latent indices that are observed to observation indices
    latent_to_obs_idxs = {}
    for i in range(len(idxs)):
        latent_to_obs_idxs[idxs[i]] = i

    
    logging.info("divided B by rowsums")
    print("divided B by rowsums")

    idxs = torch.Tensor(idxs).to(dtype=torch.int, device = 'cuda:0')

    
    B_T = B.t()
    
    

    B_TH_T_idxs = torch.stack([torch.stack([ B_T._indices()[0,i], torch.tensor(latent_to_obs_idxs[int(B_T._indices()[1,i])]).to(device='cuda:0')] ) for i in range(B_T._indices().shape[1]) if B_T._indices()[1,i] in idxs],axis=1).to(dtype=torch.int)

    B_TH_T_values = torch.stack([B_T._values()[i] for i in range(B_T._indices().shape[1]) if B_T._indices()[1,i] in idxs])
    B_TH_T = torch.sparse_coo_tensor(B_TH_T_idxs, B_TH_T_values, (int(lat*lon), y_dim ))
    
    logging.info("computed B^TH^T")
    print("computed B^TH^T")
    
    BB_TH_T = torch.sparse.mm(B, B_TH_T)
    logging.info("computed BB^TH^T")
    print("computed BB^TH^T")
    
    HBB_TH_T = torch.sparse.mm(B_TH_T.t(), B_TH_T)
    
    logging.info("computed HBB^TH^T")
    print("computed HBB^TH^T")
    
    torch.save(HBB_TH_T.to_dense().detach().cpu(), save_folder + "/HBB_TH_T{}.pt".format(tag))

    
    torch.save(BB_TH_T._values().detach().cpu(), save_folder + "/BB_TH_T_values{}.pt".format(tag))
    torch.save(BB_TH_T._indices().detach().cpu(), save_folder + "/BB_TH_T_indices{}.pt".format(tag))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=int)
    parser.add_argument("--lon", type=int)
    parser.add_argument("--factor_x", type = int, help = "factor by which the lat of the latent space is systematically subset to create observations")
    parser.add_argument("--factor_y", type = int, help = "factor by which the lon of the latent space is systematically subset to create observations")
    parser.add_argument("--tag", help = 'identifying tag for the saved files')
    parser.add_argument("--save_folder", help = "filepath where you want to save the BB^TH^T and HBB^TH^T files (must end in /)")
    args = parser.parse_args()
    # save output in a file called test.out
    logging.basicConfig(filename='2D_BB_T{}.out'.format(args.tag), encoding='utf-8', level=logging.INFO)
    
    with torch.no_grad():
                        
        lat = args.lat
        lon = args.lon 
        factor_x = args.factor_x
        factor_y = args.factor_y
        tag = args.tag
        save_folder = args.save_folder
        
        create_BB_T(lat,lon,factor_x,factor_y,tag, save_folder)