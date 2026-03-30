import re
import os
import sys
import numpy as np
from natsort import natsorted
from tqdm import tqdm

directories = ['enavg_data_time_series', 'figures_time_series']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

ncell = 100
p0 = 3.8
n_ensembles = 64

prop = ["shear_stress", "pressure" ]
idd = 0

gd_list = ["0.01", "0.001", "0.0001", "0.000025"]

for gd in gd_list:
	
	accumulated_data = None

	for en in tqdm(range(1, n_ensembles + 1), desc=f"Processing: gd={gd}"):
		if idd == 0:
			data = np.loadtxt(f"../gd_{gd}/en{en}/data/stress_strain.dat")
		elif idd == 1:
			data = np.loadtxt(f"../gd_{gd}/en{en}/data/Stress_xx_yy_strain.dat")
		
		if accumulated_data is None:
			accumulated_data = np.zeros_like(data)

		accumulated_data += data

	average_data = accumulated_data /n_ensembles

	if idd == 0:
		np.savetxt(f"{directories[0]}/average_shear_stress_p0_{p0}_gd_{gd}.dat", average_data,
	             				header="strain    stress_xy", comments='#')
	elif idd == 1:
		np.savetxt(f"{directories[0]}/average_pressure_p0_{p0}_gd_{gd}.dat", average_data,
	             				header="strain    stress_xx   stress_yy", comments='#')

print("data saved")
