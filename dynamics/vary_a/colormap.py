import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

a_values = np.logspace(-4, -1, 80)

Q_y_matrix = np.zeros((80, 80))  # Rows = a, Columns = t

ncell=100
p0 = 3.8
gd=0.001
gd_str = f"{gd:.7f}".rstrip("0").rstrip(".") 

base_dir = f"enavg_data_Qt/gd_{gd_str}"
fig_dir = f"figure_Qt"

#base_dir = f"enavg_data_corr/gd_{gd}"
#fig_dir = f"figure_corr"

os.makedirs(fig_dir, exist_ok=True) 

prop = ["qt", "chi"]
id = 1

for i, a in enumerate(a_values):
    subfolder = os.path.join(base_dir, f"a_{a:.5f}")
    # file_path = os.path.join(subfolder, f"qt_y_en_avg_a_{a:.5f}_gd_{gd}.txt")
    file_path = os.path.join(subfolder, f"{prop[id]}_y_en_avg_a_{a:.5f}_gd_{gd}.txt")
    if os.path.exists(file_path):
        data = np.loadtxt(file_path)
        time_values = data[:, 0]  # First column: time values
        Q_y_values = data[:, 1]   # Second column: Q_y values
        
        # Store Q_y values in the corresponding row
        Q_y_matrix[i, :] = Q_y_values
    else:
        print(f"Warning: Missing file for a={a:.5f}")

plt.figure(figsize=(5, 4))
plt.pcolormesh(time_values, a_values, Q_y_matrix, shading="auto", cmap="jet")

if id==0:
    plt.colorbar(label="$Q$")
elif id==1: 
    plt.colorbar(label="$\chi_4$")

plt.xlabel("$t$")
plt.ylabel("$a$")
plt.yscale("log") 
plt.xscale("log")

plt.annotate(fr"$p_0={p0}$,$\dot\gamma={gd}$", xy=(0.05, 0.95),
        xycoords='axes fraction', fontsize=10, color='white')


plt.savefig(f"{fig_dir}/color_{prop[id]}_gd_{gd}_p0_{p0}.png", dpi=200, bbox_inches='tight'
            ,pad_inches=0.01, transparent=False, facecolor='white')


