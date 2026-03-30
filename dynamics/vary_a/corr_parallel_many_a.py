import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import VmtrjReader as vt
from glob2 import glob
from natsort import natsorted
from tqdm import tqdm
from multiprocessing import Pool

ncell = 400
n_origins = 10
tf = 1e6
origin_diff = 1e6
n_ensembles = 20
gd = float(sys.argv[1])
gd_str = f"{gd:.7f}".rstrip("0").rstrip(".")
base_dir = f"enavg_data_corr/gd_{gd_str}"
os.makedirs(base_dir, exist_ok=True)  # Create base directory if not exists

# Generate 80 logarithmically spaced values of `a` from 1e-1 to 1e3
a_values = np.logspace(-4, -1, 80)

# Load common data
data_reader = vt.VmtrjReader(file=f'../../gd_{gd_str}/en1/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
print(f'Number of atoms  : {data_reader.atoms()}')
print(f'Number of frames : {data_reader.steps()[0]}')
print(f'Dimension Box    : {data_reader.boxinfo()}')

origins_files = glob(f"origin_files_{n_origins}_od_{int(origin_diff)}_tf_{int(tf)}/origin_*.txt")
origins_files = natsorted(origins_files)
all_arrays = [np.loadtxt(file) for file in origins_files]
origins = np.vstack(all_arrays)

_, frames = data_reader.steps()
steps = frames[np.where(np.isin(frames, origins[0]))[0]]

upto = 80
dt = 0.01
time = steps * dt


def process_a(a):
    subfolder = os.path.join(base_dir, f"a_{a:.5f}")  # Create a folder for each `a`
    os.makedirs(subfolder, exist_ok=True)

    ensemble_avg_qt_y = np.zeros(upto)
    ensemble_avg_chi_y = np.zeros(upto)

    for en in range(1, n_ensembles + 1):
        data_reader = vt.VmtrjReader(file=f'../../gd_{gd_str}/en{en}/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
        data = data_reader.positions(tqdm_disable=True)

        _pos = []
        for i in range(origins.shape[0]):
            indices = np.where(np.isin(frames, origins[i]))[0]
            _pos.append(data[indices])

        _pos = np.concatenate(_pos, axis=0)

        Qt_y = np.zeros(upto)
        Qt2_y = np.zeros(upto)

        for i in range(0, n_origins):
            pos = _pos[i * upto: (i + 1) * upto, :, :]

            msd_y = (pos[:, :, 1] - pos[[0], :, 1])**2

            qt_y = np.exp(-msd_y / (2 * a * a))
            qt_y = qt_y.mean(axis=1)

            Qt_y += qt_y
            Qt2_y += qt_y**2

        Qt_y /= n_origins
        Qt2_y /= n_origins

        ensemble_avg_qt_y += Qt_y
        ensemble_avg_chi_y += Qt2_y - Qt_y**2

    ensemble_avg_qt_y /= n_ensembles
    ensemble_avg_chi_y /= n_ensembles
    ensemble_avg_chi_y *= ncell

    np.savetxt(f"{subfolder}/qt_y_en_avg_a_{a:.5f}_gd_{gd_str}.txt", np.c_[time, ensemble_avg_qt_y], 
               fmt=('%16.4f', '%16.6f'), delimiter=' ', 
               header="time Qt_y")

    np.savetxt(f"{subfolder}/chi_y_en_avg_a_{a:.5f}_gd_{gd_str}.txt", np.c_[time, ensemble_avg_chi_y], 
               fmt=('%16.4f', '%16.6f'), delimiter=' ', 
               header="time Chi_y")

    print(f"Completed processing for a={a:.5f}")


if __name__ == "__main__":
    num_workers = 40  # Set number of processes to 20
    with Pool(num_workers) as p:
        p.map(process_a, a_values)

    # ** with tqdm ** progress_bar **
    # with Pool(num_workers) as p, tqdm(total=len(a_values), desc="Processing a values") as pbar:
    #     for _ in p.imap_unordered(process_a, a_values):
    #         pbar.update(1)  # Update progress bar

    print("All calculations completed for all a values!")
