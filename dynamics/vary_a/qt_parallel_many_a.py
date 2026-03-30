import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import VmtrjReader as vt
from glob2 import glob
from natsort import natsorted
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

ncell = 100
n_origins = 50
tf = 1e6
origin_diff = 1e6
n_ensembles = 64
gd = float(sys.argv[1])

gd_str = f"{gd:.7f}".rstrip("0").rstrip(".") 

base_dir = f"enavg_data_Qt/gd_{gd_str}"
os.makedirs(base_dir, exist_ok=True)  # Create base directory if not exists

a_values = np.logspace(-4, -1, 80)

data_reader = vt.VmtrjReader(file=f'../../gd_{gd_str}/en1/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
print(f'Number of atoms  : {data_reader.atoms()}')    #natoms
print(f'Number of frames : {data_reader.steps()[0]}') # nframes
print(f'Dimension Box    : {data_reader.boxinfo()}')  # t.xyz[frame,atom,coordinate] 

origins_files = glob(f"origin_files_{n_origins}_od_{int(origin_diff)}_tf_{int(tf)}/origin_*.txt")
origins_files = natsorted(origins_files)

all_arrays = [np.loadtxt(file) for file in origins_files]
origins = np.vstack(np.array(all_arrays))

_, frames = data_reader.steps()
steps = frames[np.where(np.isin(frames, origins[0]))[0]]

upto = 80
dt = 0.01
time = steps * dt

def process_a(a):
    """ Function to process a single value of `a` in parallel. """
    subfolder = os.path.join(base_dir, f"a_{a:.5f}")  # Create a folder for each `a`
    os.makedirs(subfolder, exist_ok=True)

    # ensemble_avg_qt_x = np.zeros(upto)
    ensemble_avg_qt_y = np.zeros(upto)
    # ensemble_avg_chi_x = np.zeros(upto)
    ensemble_avg_chi_y = np.zeros(upto)

    for en in range(1, n_ensembles + 1):
        data_reader = vt.VmtrjReader(file=f'../../gd_{gd_str}/en{en}/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
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
            pos = _pos[i * upto: (i + 1) * upto, :, :]

            # Self-overlap function (Qt)
            # qt_x = a - np.abs(pos[:, :, 0] - pos[[0], :, 0])
            qt_y = a - np.abs(pos[:, :, 1] - pos[[0], :, 1])

            # qt_x = np.where(qt_x > 0, 1, 0).mean(axis=1)
            qt_y = np.where(qt_y > 0, 1, 0).mean(axis=1)

            # Qt_x += qt_x
            Qt_y += qt_y

            # Four-point susceptibility (Chi4)
            # Qt2_x += qt_x**2
            Qt2_y += qt_y**2

        # ** Origin average **
        # Qt_x /= n_origins
        Qt_y /= n_origins
        # Qt2_x /= n_origins
        Qt2_y /= n_origins

        # ** Ensemble average **
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

    return f"Completed processing for a={a:.5f}"

# Parallel execution
with ProcessPoolExecutor(max_workers=32) as executor:
    results = list(tqdm(executor.map(process_a, a_values), total=len(a_values), desc="Processing a-values"))

# Print all results after completion
for res in results:
    print(res)

print("All calculations completed for all a values!")

