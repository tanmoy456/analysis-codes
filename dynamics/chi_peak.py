import re
import sys
import numpy as np
from glob2 import glob
import matplotlib.pyplot as plt
from natsort import realsorted, natsorted
from scipy.interpolate import interp1d

ncell = 100   

#a = 0.001
a = float(sys.argv[1])  # Read `a` from command-line arguments
print(f"Running chi_script with a = {a}")

p0 = 4.06
# read the files
files = glob(f"enavg_data_Qt/chi_y_en_avg_a_{a}_gd_*")
files = realsorted(files)
print("files: ", files)
# extract temp from files
def extract_float_values(file_names):
    float_values = []
    for file_name in file_names:
        match = re.search(r'gd_([\d.]+)\.txt', file_name)
        if match:
            float_value = float(match.group(1))
            float_values.append(float_value)
    return float_values

gd_list = extract_float_values(files)
gd_array = np.array(gd_list)
print("gamma dot array :", gd_array)

chi_peak = []
for file in files:
    Data = np.loadtxt(file)
    peak_value = np.max(Data[:, 1])
    chi_peak.append(peak_value)

chi_peak = np.array(chi_peak)
print("chi4_peak (Susceptibility):", chi_peak)

np.savetxt(f"enavg_data_Qt/gd_vs_chi_peak_a_{a}_ncell_{ncell}_p0_{p0}.dat", 
			np.c_[gd_array, chi_peak], 
	        fmt=('%16.8f','%16.8f'), delimiter=' ',
	        header="gd (strain rate)      chi4_peak")

plt.plot(gd_array , chi_peak,'o-', color='blue')
plt.yscale("log")
plt.xscale("log")
plt.xlabel(r"$\dot\gamma$")
plt.ylabel(r"$\chi_4^{peak}$")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.gcf().set_size_inches(5,4)
plt.savefig(f"figures_Qt/gd_vs_chi_peak_ncell_{ncell}.png",
	        dpi=200,bbox_inches='tight',pad_inches=0.05)
