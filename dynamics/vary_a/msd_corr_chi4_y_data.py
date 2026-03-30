import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import VmtrjReader as vt
from glob2 import glob
from natsort import natsorted
from tqdm import tqdm

ncell = 400
n_origins = 10
tf = 1e6
origin_diff = 1e6
n_ensembles = 40
gd = float(sys.argv[1])

base_dir = f"enavg_data_corr/gd_{gd}"
os.makedirs(base_dir, exist_ok=True)  # Create base directory if not exists

# Generate 80 logarithmically spaced values of `a` from 1e-1 to 1e3
a_values = np.logspace(-1, 3, 80)

data_reader = vt.VmtrjReader(file=f'../../gd_{gd}/en1/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
print(f'Number of atoms  : {data_reader.atoms()}')    #natoms
print(f'Number of frames : {data_reader.steps()[0]}') # nframes
print(f'Dimension Box    : {data_reader.boxinfo()}')  # t.xyz[frame,atom,coordinate] 

origins_files = glob(f"origin_files_{n_origins}_od_{int(origin_diff)}_tf_{int(tf)}/origin_*.txt")
origins_files = natsorted(origins_files)
all_arrays = [np.loadtxt(file) for file in origins_files]
origins = np.vstack(all_arrays)

_, frames = vt.VmtrjReader(file=f'../../gd_{gd}/en1/data/Com_All_Cells_Unwrapped_N_{ncell}.dat').steps()
steps = frames[np.where(np.isin(frames, origins[0]))[0]]

upto = 80
dt = 0.01
time = steps * dt

# Loop over all values of `a`
for a in a_values:
    subfolder = os.path.join(base_dir, f"a_{a:.5f}")  # Create a folder for each `a`
    os.makedirs(subfolder, exist_ok=True)

    # ensemble_avg_qt_x = np.zeros(upto)  
    ensemble_avg_qt_y = np.zeros(upto)  
    # ensemble_avg_chi_x = np.zeros(upto)  
    ensemble_avg_chi_y = np.zeros(upto)  

    for en in tqdm(range(1, n_ensembles + 1), desc=f"Processing a={a:.5f}, gd={gd}"):
        data_reader = vt.VmtrjReader(file=f'../../gd_{gd}/en{en}/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
        data = data_reader.positions(tqdm_disable=True)

        _pos = []
        for i in range(origins.shape[0]):
            indices = np.where(np.isin(frames, origins[i]))[0]
            _pos.append(data[indices])

        _pos = np.concatenate(_pos, axis=0)

        # Qt_x = np.zeros(upto)
        Qt_y = np.zeros(upto)
        # Qt2_x = np.zeros(upto)
        Qt2_y = np.zeros(upto)

        for i in range(0, n_origins):
            pos = _pos[i * upto: (i + 1) * upto, :, :]  # [upto, cell, 2]

            # Mean Square Displacement (MSD)
            # msd_x = (pos[:, :, 0] - pos[[0], :, 0])**2  # x-component
            msd_y = (pos[:, :, 1] - pos[[0], :, 1])**2  # y-component

            # Correlation function (Qs)
            # qt_x = np.exp(-msd_x / (2 * a * a))
            qt_y = np.exp(-msd_y / (2 * a * a))
            
            # qt_x = qt_x.mean(axis=1)  # mean over ncell 
            qt_y = qt_y.mean(axis=1)

            # Qt_x += qt_x  # These now have shape (upto,)
            Qt_y += qt_y

            # Four-point susceptibility (Chi4)
            # Qt2_x += qt_x**2
            Qt2_y += qt_y**2

        # Origin average
        # Qt_x /= n_origins
        Qt_y /= n_origins
        # Qt2_x /= n_origins
        Qt2_y /= n_origins

        # Ensemble average
        # ensemble_avg_qt_x += Qt_x
        ensemble_avg_qt_y += Qt_y
        # ensemble_avg_chi_x += Qt2_x - Qt_x**2
        ensemble_avg_chi_y += Qt2_y - Qt_y**2

    # ensemble_avg_qt_x /= n_ensembles
    ensemble_avg_qt_y /= n_ensembles
    # ensemble_avg_chi_x /= n_ensembles
    ensemble_avg_chi_y /= n_ensembles

    # ensemble_avg_chi_x *= ncell
    ensemble_avg_chi_y *= ncell

    # Save results in corresponding subfolder
    # np.savetxt(f"{subfolder}/qt_x_en_avg_a_{a:.5f}_gd_{gd}.txt", np.c_[time, ensemble_avg_qt_x], fmt=('%16.4f', '%16.6f'), delimiter=' ')
    np.savetxt(f"{subfolder}/qt_y_en_avg_a_{a:.5f}_gd_{gd}.txt", np.c_[time, ensemble_avg_qt_y], fmt=('%16.4f', '%16.6f'), delimiter=' ')
    # np.savetxt(f"{subfolder}/chi_x_en_avg_a_{a:.5f}_gd_{gd}.txt", np.c_[time, ensemble_avg_chi_x], fmt=('%16.4f', '%16.6f'), delimiter=' ')
    np.savetxt(f"{subfolder}/chi_y_en_avg_a_{a:.5f}_gd_{gd}.txt", np.c_[time, ensemble_avg_chi_y], fmt=('%16.4f', '%16.6f'), delimiter=' ')

    print(f"Completed processing for a={a:.5f}")

print("All calculations completed for all a values!")
