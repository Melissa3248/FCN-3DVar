import torch
import matplotlib.pyplot as plt
import numpy as np

offset = 0
half_window = 50
m=np.load('../../stats_v0/global_means.npy')
s=np.load('../../stats_v0/global_stds.npy')

max_t = 41
N_ensem = 50

device = 'cuda:0'

def get_eye_over_t(dataset, m,s, save_dir):

    min_p_over_ens_and_t = []
    
    min_p = torch.empty((N_ensem,max_t)) # n_ensem , n_tpts
    max_ws = torch.empty((N_ensem,max_t)) # n_ensem , n_tpts

    for n in range(N_ensem):
        min_pressure_over_t = []
        for i in range(max_t):
            if i <=7: # hand chosen to capture the eye of the typhoon (other low pressure/high wind regions that aren't the typhoon are cut)
                offset = 75
            else: 
                offset = 0
            print(n,i)
            pressure = np.load(save_dir + "/{}_preds{}.npy".format(dataset,i))[n,4,200:350,450+15:600+15]
            s_v10 = s[0,1,0,0]; m_v10 = m[0,1,0,0]
            s_u10 = s[0,0,0,0]; m_u10 = m[0,0,0,0]
            s_p = s[0,4,0,0]; m_p = m[0,4,0,0]
            
            pressure = pressure*s_p + m_p
            pressure = torch.from_numpy(pressure).to(device)
            
            v10 = np.load(save_dir + "/{}_preds{}.npy".format(dataset,i))[n,1,200:350,450+15:600+15]#200:400,450:650]
            v10 = v10*s_v10 + m_v10
            v10 = torch.from_numpy(v10).to(device)

            u10 = np.load(save_dir + "/{}_preds{}.npy".format(dataset,i))[n,0,200:350,450+15:600+15]
            u10 = u10*s_u10 + m_u10
            u10 = torch.from_numpy(u10).to(device)

            ws = torch.sqrt(u10*u10 + v10*v10)

            max_ws[n,i] = torch.max(ws)

            min_pressure = (pressure[offset:,:]==torch.min(pressure[offset:,:])).nonzero()

            
            if len(min_pressure) >1:
                try:
                    print("more than 1 {}".format(dataset)) # more than one minimum pressure location was identified
                except:
                    pass
                min_pressure = min_pressure[0] # just choose the first one
            
            min_p[n,i] = torch.min(pressure[offset:,:])

            min_pressure_over_t.append([min_pressure.squeeze()[1], min_pressure.squeeze()[0]+offset])

        min_p_over_ens_and_t.append(min_pressure_over_t)
    
    return torch.tensor(min_p_over_ens_and_t).detach().cpu().numpy(), min_p, max_ws

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor",type=int)
    parser.add_argument("--dataset", type = str, help = "options: clean, obs, analysis")
    parser.add_argument("--save_dir", type = str)
    args = parser.parse_args()
    
    factor = args.factor
    
    dataset = args.dataset 
    save_dir = args.save_dir.format(factor)
    res, min_p, max_ws = get_eye_over_t(dataset, m, s, save_dir)
    
    np.save(save_dir + "/{}_res.npy".format( dataset), res)
    np.save(save_dir + "/{}_min_p.npy".format(dataset), min_p)
    np.save(save_dir + "/{}_max_ws.npy".format(dataset), max_ws)