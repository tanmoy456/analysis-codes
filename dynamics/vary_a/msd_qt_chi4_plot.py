import re
import os
import sys
import numpy as np
from glob2 import glob
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from natsort import realsorted, natsorted

# directories = ['enavg_data_Qt', 'figures_Qt']
# directories = ['enavg_data_corr', 'figures_corr']
directories = ['enavg_data_fs', 'figures_fs']

ncell = 100
T = 0.0
p0_values = 3.8
a = 0.01
prop = ["msd_y", "qt_y", "chi_y"]
gd_values = ["0.01","0.005", "0.001", "0.0001"]

markers = ['o', 's', '^', 'D', 'p', 'H', '*', '+', 'x', '1', '2', '3',]

for id in tqdm(range(len(prop)), desc="Processing properties"):
    prop_name = prop[id]
    plt.figure(figsize=(8, 6))
    marker_idx = 0 
    for gd in gd_values:
        base_dir = f"{directories[0]}/" + prop_name + f"_en_avg_a_{a}_gd_{gd}.txt"
        label = fr"$\dot\gamma={gd}$"
        try:
            data = np.loadtxt(base_dir)
            time, values = data[:, 0], data[:, 1]
            plt.plot(time, values, marker=markers[marker_idx % len(markers)], linestyle='-', label=label)
            marker_idx += 1
        except Exception as e:
                print(f"Error loading {base_dir}: {e}")

    plt.xscale("log")
    plt.xlabel(r"$t$")

    if id == 0:
        plt.yscale("log")
        plt.ylabel(r"$MSD^y(t)$")
    elif id == 1:
        plt.ylabel(r"$Q^y(t)$")
    elif id == 2:
        plt.ylabel(r"$\chi_{4}^{y}(t)$")

    plt.legend(fontsize=8)  # Smaller legend for clarity
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Save the figure
    plt.savefig(f"{directories[1]}/{prop_name}.png", dpi=200,
                bbox_inches='tight',pad_inches=0.01, 
                transparent=False, facecolor='white')
    plt.close()  # Close figure to avoid overlap

print("Comparison plots saved successfully!")
