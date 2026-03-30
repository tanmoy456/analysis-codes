import re
import sys
import numpy as np
from glob2 import glob
import matplotlib.pyplot as plt
from natsort import realsorted, natsorted
from scipy.interpolate import interp1d

# interpolation 
def reverse_linear_interpolation(data_points, y_target):
    above_0_5 = None
    below_0_5 = None

    for point in data_points:
        x, y = point[0], point[1]
        if y > y_target:
            if above_0_5 is None or y < above_0_5[1]:
                above_0_5 = (x, y)
        elif y < y_target:
            if below_0_5 is None or y > below_0_5[1]:
                below_0_5 = (x, y)

    x_values = [above_0_5[0], below_0_5[0]] if above_0_5 and below_0_5 else []
    y_values = [above_0_5[1], below_0_5[1]] if above_0_5 and below_0_5 else []

    if len(x_values) != 2 or len(y_values) != 2:
        return None  # Insufficient data points for interpolation

    x_target = x_values[0] + (x_values[1] - x_values[0]) * (y_target - y_values[0]) / (y_values[1] - y_values[0])
    return x_target

p0=4.06
ncell = 100                 # system size
y_target = 1/np.exp(1)  # Target y-value
#y_target = 0.5
print("Q(t) at ", y_target)
#a = 0.001
a = float(sys.argv[1])  # Read `a` from command-line arguments
print(f"Running tau_script with a = {a}")

# read the files
files = glob(f"enavg_data_Qt/qt_y_en_avg_a_{a}_gd_*")
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

tau = []
for file in files:
    Data = np.loadtxt(file) 
    x_target = reverse_linear_interpolation(Data, y_target)
    tau.append(x_target)
tau_array = np.array(tau)  
print("tau_alph :", tau_array)

# save data in a file
np.savetxt(f"enavg_data_Qt/gd_vs_tau_alpha_a_{a}_ncell_{ncell}_p0_{p0}.dat",
	np.c_[gd_array, tau_array], fmt=('%16.8f','%16.8f'), delimiter=' ',
    header="gd (strain rate)      tau_alpha")

plt.plot(gd_array , tau_array,'o-', color='blue')
plt.yscale("log")
plt.xscale("log")
plt.xlabel(r"$\dot\gamma$")
plt.ylabel(r"$\tau_{\alpha}$")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
plt.gcf().set_size_inches(5,4)
plt.savefig(f"figures_Qt/gd_vs_tau_alpha_ncell_{ncell}.png",
	        dpi=200,bbox_inches='tight',pad_inches=0.05)
