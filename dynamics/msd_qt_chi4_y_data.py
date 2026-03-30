import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import VmtrjReader as vt
from glob2 import glob
from natsort import natsorted
from tqdm import tqdm

# create directories if not there
directories = ['enavg_data_Qt', 'figures_Qt']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

ncell = 100
n_origins = 10
tf = 1e6
origin_diff = 1e6
n_ensembles = 16
gd =  float(sys.argv[1])

gd_str = f"{gd:.7f}".rstrip("0").rstrip(".")

print("gamma_dot :", gd)

a = 0.004

data_reader = vt.VmtrjReader(file=f'../gd_{gd_str}/en1/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
print(f'Number of atoms  : {data_reader.atoms()}')    #natoms
print(f'Number of frames : {data_reader.steps()[0]}') # nframes
print(f'Dimension Box    : {data_reader.boxinfo()}')  # t.xyz[frame,atom,coordinate] 

origins_files = glob(f"origin_files_{n_origins}_od_{int(origin_diff)}_tf_{int(tf)}/origin_*.txt")
origins_files = natsorted(origins_files ) 

all_arrays = []
for file in origins_files :
    array = np.loadtxt(file)
    all_arrays.append(array)
# Concatenate the arrays vertically
origins = np.vstack(np.array(all_arrays))

_, frames = data_reader.steps()
steps = frames[np.where(np.isin(frames, origins[0]))[0]]

upto = 80
dt = 0.01
time = (steps-1) * dt

ensemble_avg_msd_x = np.zeros(upto)  
ensemble_avg_msd_y = np.zeros(upto)  
ensemble_avg_qt_x = np.zeros(upto)  
ensemble_avg_qt_y = np.zeros(upto)  
ensemble_avg_chi_x = np.zeros(upto)  
ensemble_avg_chi_y = np.zeros(upto)  

for en in tqdm(range(1, n_ensembles + 1), desc=f"Processing: gd={gd}"):
    data_reader = vt.VmtrjReader(file=f'../gd_{gd_str}/en{en}/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
    data = data_reader.positions(tqdm_disable=True)

    _pos = []
    for i in range(origins.shape[0]):
        indices = np.where(np.isin(frames, origins[i]))[0]
        _pos.append(data[indices])

    _pos = np.concatenate(_pos, axis=0)

    MSD_x = np.zeros(upto)
    MSD_y = np.zeros(upto)
    Qt_x = np.zeros(upto)
    Qt_y = np.zeros(upto)
    Qt2_x = np.zeros(upto)
    Qt2_y = np.zeros(upto)

    for i in range(0, n_origins):
        pos = _pos[i * upto: (i + 1) * upto, :, :]

        # Mean Square Displacement (MSD)
        msd_x = (pos[:, :, 0] - pos[[0], :, 0])**2  # x-component
        msd_y = (pos[:, :, 1] - pos[[0], :, 1])**2  # y-component
        MSD_x += msd_x.mean(axis=-1)
        MSD_y += msd_y.mean(axis=-1)

        # Self-overlap function (Qt)
        qt_x = a - np.abs(pos[:, :, 0] - pos[[0], :, 0])
        qt_y = a - np.abs(pos[:, :, 1] - pos[[0], :, 1])
        
        qt_x = np.where(qt_x > 0, 1, 0).mean(axis=1)
        qt_y = np.where(qt_y > 0, 1, 0).mean(axis=1)
        
        Qt_x += qt_x
        Qt_y += qt_y

        # Four-point susceptibility (Chi4)
        Qt2_x += qt_x**2
        Qt2_y += qt_y**2

    # Origin average
    MSD_x /= n_origins
    MSD_y /= n_origins
    Qt_x /= n_origins
    Qt_y /= n_origins
    Qt2_x /= n_origins
    Qt2_y /= n_origins

    # Ensemble average
    ensemble_avg_msd_x += MSD_x
    ensemble_avg_msd_y += MSD_y
    ensemble_avg_qt_x += Qt_x
    ensemble_avg_qt_y += Qt_y
    ensemble_avg_chi_x += Qt2_x - Qt_x**2
    ensemble_avg_chi_y += Qt2_y - Qt_y**2

ensemble_avg_msd_x /= n_ensembles
ensemble_avg_msd_y /= n_ensembles
ensemble_avg_qt_x /= n_ensembles
ensemble_avg_qt_y /= n_ensembles
ensemble_avg_chi_x /= n_ensembles
ensemble_avg_chi_y /= n_ensembles

ensemble_avg_chi_x *= ncell
ensemble_avg_chi_y *= ncell

# Save the data
# np.savetxt(f"{directories[0]}/msd_x_en_avg_a_{a}_gd_{gd}.txt", np.c_[time, ensemble_avg_msd_x], fmt=('%16.4f', '%16.6f'), delimiter=' ')
np.savetxt(f"{directories[0]}/msd_y_en_avg_gd_{gd_str}.txt", np.c_[time, ensemble_avg_msd_y], fmt=('%16.8f', '%16.8f'), delimiter=' ')
# np.savetxt(f"{directories[0]}/qt_x_en_avg_a_{a}_gd_{gd}.txt", np.c_[time, ensemble_avg_qt_x], fmt=('%16.4f', '%16.6f'), delimiter=' ')
np.savetxt(f"{directories[0]}/qt_y_en_avg_a_{a}_gd_{gd_str}.txt", np.c_[time, ensemble_avg_qt_y], fmt=('%16.8f', '%16.8f'), delimiter=' ')
# np.savetxt(f"{directories[0]}/chi_x_en_avg_a_{a}_gd_{gd}.txt", np.c_[time, ensemble_avg_chi_x], fmt=('%16.4f', '%16.6f'), delimiter=' ')
np.savetxt(f"{directories[0]}/chi_y_en_avg_a_{a}_gd_{gd_str}.txt", np.c_[time, ensemble_avg_chi_y], fmt=('%16.8f', '%16.8f'), delimiter=' ')

print("MSD_x, MSD_y, Qt_x, Qt_y, Chi_x, Chi_y: done!")
