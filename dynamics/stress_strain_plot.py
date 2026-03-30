import re
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm

directories = ['enavg_data_time_series', 'figures_time_series']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

ncell = 100
p0 = 3.8

prop = ["shear_stress", "pressure" ]
#idd = 0

gd_list = ["0.01", "0.001", "0.0001", "0.000025"]

plt.figure(figsize=(6, 4))

for gd in gd_list:

	data1 = np.loadtxt(f"{directories[0]}/average_{prop[0]}_p0_{p0}_gd_{gd}.dat")
	#data2 = np.loadtxt(f"{directories[0]}/average_{prop[1]}_p0_{p0}_gd_{gd}.dat")

	plt.plot(data1[:,0], data1[:,1], linestyle='-', 
    		    label=fr"$\dot{{\gamma}}$ = {gd}, $\Sigma_{{xy}}$")
	# plt.plot(data2[:,0], data2[:,1], label=fr"$\dot{{\gamma}}$ = {gd}, $\Sigma_{{xx}}$")
	# plt.plot(data2[:,0], data2[:,2], label=fr"$\dot{{\gamma}}$ = {gd}, $\Sigma_{{yy}}$")
	#plt.plot(data2[:,0], (data2[:,1]+data2[:,2])/2, linestyle='--',
	#        label=fr"$\dot{{\gamma}}$ = {gd}, $P$")


plt.xlim(0,10)
plt.ylim(0.0, None)
plt.legend(ncol=2)
plt.ylabel(fr"$\Sigma_{{xy}}$ / $P$")
plt.xlabel(fr"$\gamma$")
plt.savefig(f'{directories[1]}/stress_strain_p0_{p0}.png', dpi= 200,
             bbox_inches='tight',pad_inches=0.01, transparent=False)
