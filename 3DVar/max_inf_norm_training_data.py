import h5py
import numpy as np

import time

import random
random.seed(10)

# initialize the maximum inf norms for each feature
max_norms = np.repeat(0, 20)

data_path = '/pscratch/sd/m/maadrian/data/ERA5/train/'

years = np.random.choice(range(1979, 2016), size=5, replace=False)
print(years)

for y in years:
    start = time.time()
    year = h5py.File(data_path + "{}.h5".format(y))['fields'][:,:20,:,:]
    
    # matrix of size (tpts, features) with inf norms
    norms_over_time =np.linalg.norm(year.reshape(year.shape[0],year.shape[1],year.shape[2]*year.shape[3]), ord = np.inf, axis = 2) 
    
    # take max of inf norms across time
    norms = np.max(norms_over_time, axis=0)
    
    print(norms.shape)
    max_norms = np.max(np.stack((norms,max_norms)), axis =0)
    

    end = time.time()
    print(y, end - start)
    
np.save("max_inf_norms.npy", max_norms)