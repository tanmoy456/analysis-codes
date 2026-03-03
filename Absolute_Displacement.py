import os
import numpy as np
from glob2 import glob
from natsort import natsorted
from tqdm import tqdm
import VmtrjReader as vt

# -----------------------------
# Parameters
# -----------------------------
p0 = 4.2
ncell = 400
n_ensembles = 128
n_origins = 1
origin_diff = 1e6
tf = 1e6

upto = 80
dt = 0.01
L = np.sqrt(ncell)          # displacement cutoff

v0_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1,
           0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]

# -----------------------------
# Load origins
# -----------------------------
origin_dir = f"origin_files_{n_origins}_od_{int(origin_diff)}_tf_{int(tf)}"
origin_files = glob(f"{origin_dir}/origin_*.txt")
origin_files = natsorted(origin_files)

if len(origin_files) == 0:
    raise RuntimeError("No origin files found.")

origins = np.vstack([np.loadtxt(f) for f in origin_files])

# -----------------------------
# Prepare output
# -----------------------------
os.makedirs("enavg_data", exist_ok=True)
jammed_results = []   # store (v0, jammed_fraction)

# =============================
# Loop over v0
# =============================
for v0 in tqdm(v0_list):

    print(f"\n==============================")
    print(f"Processing v0 = {v0}")
    print(f"==============================")

    # Check if first ensemble exists
    first_file = f'../v0_{v0}/en1/data/Com_All_Cells_Unwrapped_N_{ncell}.dat'
    if not os.path.exists(first_file):
        print(f"Skipping v0={v0} (no en1 found)")
        continue

    # -----------------------------
    # Read frame indices
    # -----------------------------
    try:
        reader0 = vt.VmtrjReader(file=first_file)
        _, frames = reader0.steps()
    except Exception as e:
        print(f"Skipping v0={v0} (cannot read frames: {e})")
        continue

    steps = frames[np.where(np.isin(frames, origins[0]))[0]]

    if len(steps) < upto:
        print(f"Skipping v0={v0} (not enough frames)")
        continue

    time = (steps[:upto] - 1) * dt

    # -----------------------------
    # Storage
    # -----------------------------
    ensemble_avg_Dr = np.zeros(upto)
    flowing_count = 0
    valid_ensembles = 0
    failed_ensembles = []

    # =============================
    # Ensemble loop
    # =============================
    for en in range(1, n_ensembles + 1):

        file_path = f'../v0_{v0}/en{en}/data/Com_All_Cells_Unwrapped_N_{ncell}.dat'

        if not os.path.exists(file_path):
            failed_ensembles.append(en)
            continue

        try:
            reader = vt.VmtrjReader(file=file_path)
            data = reader.positions(tqdm_disable=True)
        except Exception as e:
            print(f"Skipping en{en} (error: {e})")
            failed_ensembles.append(en)
            continue

        # -------------------------
        # Collect origin trajectories
        # -------------------------
        pos_all = []
        for i in range(origins.shape[0]):
            idx = np.where(np.isin(frames, origins[i]))[0]
            if len(idx) < upto:
                continue
            pos_all.append(data[idx[:upto]])

        if len(pos_all) == 0:
            failed_ensembles.append(en)
            continue

        pos_all = np.concatenate(pos_all, axis=0)

        Dr = np.zeros(upto)

        for i in range(n_origins):
            pos = pos_all[i*upto:(i+1)*upto]

            if pos.shape[0] < upto:
                continue

            disp = np.linalg.norm(pos - pos[0], axis=-1)
            Delta_r = disp.mean(axis=1)
            Dr += Delta_r

        Dr /= n_origins

        ensemble_avg_Dr += Dr

        if np.max(Dr) >= L:
            flowing_count += 1

        valid_ensembles += 1

    # -----------------------------
    # Skip v0 if no valid ensembles
    # -----------------------------
    if valid_ensembles == 0:
        print(f"No valid ensembles for v0={v0}. Skipping.")
        continue

    # -----------------------------
    # Ensemble averages
    # -----------------------------
    ensemble_avg_Dr /= valid_ensembles

    flowing_fraction = flowing_count / valid_ensembles
    jammed_fraction = 1.0 - flowing_fraction

    jammed_results.append([v0, jammed_fraction])

    # -----------------------------
    # Save Δr(t)
    # -----------------------------
    np.savetxt(
        f"enavg_data/Delta_r_en_avg_v0_{v0}.txt",
        np.c_[time, ensemble_avg_Dr],
        fmt=('%16.6f', '%16.6e'),
        header="time  Delta_r(t)"
    )

    print(f"\nSummary for v0 = {v0}")
    print(f"Valid ensembles   = {valid_ensembles}")
    print(f"Failed ensembles  = {failed_ensembles}")
    print(f"Flowing fraction  = {flowing_fraction:.3f}")
    print(f"Jammed fraction   = {jammed_fraction:.3f}")

# -----------------------------
# Save jammed fraction vs v0
# -----------------------------
if len(jammed_results) > 0:
    jammed_results = np.array(jammed_results)

    np.savetxt(
        f"enavg_data/jammed_fraction_vs_v0_p0_{p0}.txt",
        jammed_results,
        fmt=('%10.4f', '%10.6f'),
        header="v0  jammed_fraction"
    )

    print("\nSaved jammed_fraction_vs_v0 file.")
else:
    print("\nNo valid v0 processed.")

print("\nAll processing completed.")
