import numpy as np
import os
from tqdm import tqdm

# ==========================================================
# Function: compute velocity statistics for one ensemble
# ==========================================================
def velocity_stats(filename, t_start):

    step_means = []
    step_numbers = []
    all_speeds = []

    try:
        with open(filename, 'r') as f:

            vx_list = []
            vy_list = []
            current_step = None

            for line in f:
                line = line.strip()

                if line.startswith("STEP"):

                    if vx_list:
                        vx = np.array(vx_list)
                        vy = np.array(vy_list)
                        speed = np.sqrt(vx**2 + vy**2)

                        step_means.append(speed.mean())
                        step_numbers.append(current_step)

                        if current_step >= t_start:
                            all_speeds.extend(speed)

                        vx_list = []
                        vy_list = []

                    current_step = int(line.split(":")[1])
                    continue

                if line.startswith("Strain") or not line:
                    continue

                parts = line.split()
                if len(parts) == 6:
                    vx_list.append(float(parts[2]))
                    vy_list.append(float(parts[3]))

            # last block
            if vx_list:
                vx = np.array(vx_list)
                vy = np.array(vy_list)
                speed = np.sqrt(vx**2 + vy**2)

                step_means.append(speed.mean())
                step_numbers.append(current_step)

                if current_step >= t_start:
                    all_speeds.extend(speed)

    except Exception as e:
        return None, None

    step_means = np.array(step_means)
    step_numbers = np.array(step_numbers)
    all_speeds = np.array(all_speeds)

    if len(step_numbers) == 0:
        return None, None

    mask = step_numbers >= t_start
    if not np.any(mask):
        return None, None

    steady_mean = step_means[mask].mean()

    return steady_mean, all_speeds


# ===============
# Main parameters
# ===============
p0 = 4.2

v0_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1,
           0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]

n_ensembles = 96
t_start = 2e4

output_dir = "velocity_analysis"
os.makedirs(output_dir, exist_ok=True)

v0_vs_avg_velocity = []

# ============
# Loop over v0
# ============
for v0 in v0_list:

    print("\n====================================")
    print(f"Processing v0 = {v0}")
    print("====================================")

    ensemble_means = []
    global_distribution = []
    failed_ensembles = []
    valid_ensembles = 0

    for en in tqdm(range(1, n_ensembles + 1), desc=f"v0={v0}"):

        filename = f"../v0_{v0}/en{en}/data/velocity.dat"

        if not os.path.exists(filename):
            failed_ensembles.append(en)
            continue

        mean_v, speeds = velocity_stats(filename, t_start)

        if mean_v is None:
            failed_ensembles.append(en)
            continue

        ensemble_means.append(mean_v)
        global_distribution.extend(speeds)
        valid_ensembles += 1

    # ==========================
    # Skip v0 if no valid data
    # ==========================
    if valid_ensembles == 0:
        print(f"No valid ensembles for v0={v0}. Skipping.")
        continue

    ensemble_means = np.array(ensemble_means)
    global_distribution = np.array(global_distribution)

    ensemble_avg = np.mean(ensemble_means)
    v0_vs_avg_velocity.append([v0, ensemble_avg])

    print(f"Valid ensembles   = {valid_ensembles}")
    print(f"Failed ensembles  = {failed_ensembles}")
    print(f"Ensemble average velocity = {ensemble_avg}")
    print(f"Total speeds stored = {len(global_distribution)}")

    # =============================
    # Save distribution
    # =============================
    if len(global_distribution) > 0:

        vmin = max(global_distribution.min(), 1e-6)
        vmax = global_distribution.max()

        if vmax > vmin:
            bins = np.geomspace(vmin, vmax, 80)

            hist, edges = np.histogram(global_distribution,
                                       bins=bins,
                                       density=False)

            centers = (edges[:-1] + edges[1:]) / 2

            dist_data = np.column_stack((centers, hist))

            np.savetxt(
                os.path.join(output_dir,
                             f"velocity_distribution_v0_{v0}.dat"),
                dist_data,
                header="velocity  counts",
                fmt="%.6e"
            )

# ===========================
# Save v0 vs average velocity
# ===========================
if len(v0_vs_avg_velocity) > 0:

    v0_vs_avg_velocity = np.array(v0_vs_avg_velocity)

    np.savetxt(
        os.path.join(output_dir,
                     f"v0_vs_avg_velocity_p0_{p0}.dat"),
        v0_vs_avg_velocity,
        header="v0  ensemble_avg_velocity",
        fmt="%.6e"
    )

    print("\nSaved v0_vs_avg_velocity file.")
else:
    print("\nNo valid v0 processed.")

print("\nAll processing completed.")

