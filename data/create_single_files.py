import h5py
import numpy as np

data = h5py.File("2023.h5", 'r')['fields']

# save files as individual files (.npy)
for t in range(data.shape[0]):
    one_t = data[t,:,:,:]

    # add dimension to initial condition
    one_t = np.expand_dims(one_t, axis=0)

    np.save(f"individual_files/2023_t{t}.npy", one_t)