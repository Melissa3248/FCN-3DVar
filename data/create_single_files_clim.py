import h5py
import numpy as np
import torch
device = 'cuda:0'

years = range(1979,2016)#[1979, 1989, 1999, 2004, 2010]
for t in range(1460):
    clim = []
    for y in years:
        print(y,t)
        clim.append(torch.tensor(h5py.File(f"ERA5/{y}.h5", 'r')['fields'][t,:20,:720,:1440]).to(device).unsqueeze(0))
    clim = torch.stack(clim,axis = 0).mean(0)
    np.save("individual_files_clim/climatology_t{}.npy".format(t), clim.detach().cpu().numpy())
    
