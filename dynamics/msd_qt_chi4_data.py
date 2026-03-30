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
directories = ['enavg_data', 'figures']

for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")
        
ncell = 100
n_origins = 50
origin_diff = 1e5
n_ensembles = 20

temp =  float(sys.argv[1])
#v0 = 0.3
#temp = float(input("Enter the value of v0: "))
print("temp :", temp)

data_reader = vt.VmtrjReader(file=f'../temp_{temp}/en1/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
print(f'Number of atoms  : {data_reader.atoms()}')    #natoms
print(f'Number of frames : {data_reader.steps()[0]}') # nframes
print(f'Dimension Box    : {data_reader.boxinfo()}')  # t.xyz[frame,atom,coordinate] 

#origin_files_100
origins_files = glob(f"origin_files_{n_origins}_od_{int(origin_diff)}/origin_*.txt")
origins_files = natsorted(origins_files ) 
#print("origin files",origins_files[0:5])
all_arrays = []
for file in origins_files :
    array = np.loadtxt(file)
    all_arrays.append(array)
# Concatenate the arrays vertically
origins = np.vstack(np.array(all_arrays))

_, frames = data_reader.steps()
steps = frames[np.where(np.isin(frames, origins[0]))[0]]
#print(len(steps))

#n_ensembles = 32
upto = 80
dt = 0.01
time = steps * dt

#a = np.sqrt(0.12)
a = np.sqrt(0.12)

ensemble_avg_msd = np.zeros(upto)
ensemble_avg_qt = np.zeros(upto)   
ensemble_avg_chi = np.zeros(upto)

for en in tqdm(range(1, n_ensembles  + 1)):
    data_reader = vt.VmtrjReader(file=f'../temp_{temp}/en{en}/data/Com_All_Cells_Unwrapped_N_{ncell}.dat')
    data = data_reader.positions(tqdm_disable=True)

    _pos = []
    for i in range(origins.shape[0]):
        indices = np.where(np.isin(frames, origins[i]))[0]
        _pos.append(data[indices])

    _pos = np.concatenate(_pos, axis=0)

    MSD = np.zeros(upto)
    Qt = np.zeros(upto)
    Qt2 = np.zeros(upto)

    for i in range(0, n_origins):
        pos = _pos[(i)*upto : (upto)*(i+1), :, :]

        # mean squre displacement
        msd = np.linalg.norm(pos - pos[[0], :, :], axis=-1)**2
        MSD += msd.mean(axis=-1)
        
        # self overlap function
        qt = a - np.linalg.norm(pos - pos[[0], :, :], axis=-1)
        qt = np.where(qt > 0, 1, 0)
        qt = np.mean(qt, axis=1)
        Qt += qt
        
        # four point susceptibility
        qt2 = qt**2
        Qt2 += qt2
        
    # origin avarage
    MSD /= n_origins
    Qt /= n_origins
    Qt2 /= n_origins
    # ensemble avarage
    ensemble_avg_msd += MSD
    ensemble_avg_qt += Qt
    ensemble_avg_chi += Qt2 - Qt**2
    
ensemble_avg_msd /= n_ensembles 
ensemble_avg_qt /= n_ensembles 
ensemble_avg_chi /= n_ensembles

ensemble_avg_chi *= ncell

np.savetxt(f"enavg_data/msd_en_avg_T_{temp}.txt", np.c_[time, ensemble_avg_msd], fmt=('%16.4f','%16.6f'), delimiter=' ')
np.savetxt(f"enavg_data/qt_en_avg_T_{temp}.txt", np.c_[time, ensemble_avg_qt], fmt=('%16.4f','%16.6f'), delimiter=' ')
np.savetxt(f"enavg_data/chi_en_avg_T_{temp}.txt", np.c_[time, ensemble_avg_chi], fmt=('%16.4f','%16.6f'), delimiter=' ')

print("msd, qt, chi4 ==> done!")
