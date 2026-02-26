import os
from glob import glob

import numpy as np
from tqdm import tqdm


# ==========================================================
# Parameters
# ==========================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UPTO = 80
DT = 0.01
N_ORIGINS = 1
ORIGIN_DIFF = 1e6
TF = 1e6


# ==========================================================
# Utility Functions
# ==========================================================

def numeric_sort(folder_list, prefix):
    return sorted(folder_list, key=lambda x: float(x.split(prefix)[1]))


def build_analysis_root(p0_path, search_path):
    sim_name = os.path.basename(search_path.rstrip(os.sep))
    parts = sim_name.split("_")

    kv = {}
    if len(parts) % 2 == 0:
        for i in range(0, len(parts), 2):
            kv[parts[i]] = parts[i + 1]

    if "Dr" in kv and "T" in kv:
        analysis_name = f"analysis_Dr_{kv['Dr']}_T_{kv['T']}"
    else:
        analysis_name = f"analysis_{sim_name}"

    return os.path.join(p0_path, analysis_name)


def read_origin_steps(origin_dir, n_origins):
    origin_files = sorted(glob(os.path.join(origin_dir, "origin_*.txt")))
    if not origin_files:
        raise FileNotFoundError(f"No origin_*.txt found in {origin_dir}")

    if n_origins is not None:
        origin_files = origin_files[:n_origins]

    origin_steps = []
    for file_path in origin_files:
        vals = np.loadtxt(file_path, dtype=int)
        vals = np.atleast_1d(vals).astype(int)
        origin_steps.append(vals.tolist())

    return origin_steps


def resolve_origin_dir(sim_path, n_origins, origin_diff, tf):
    analysis_dir = os.path.join(sim_path, "analysis")
    if not os.path.isdir(analysis_dir):
        return None

    preferred = os.path.join(
        analysis_dir,
        f"origin_files_{n_origins}_od_{int(origin_diff)}_tf_{int(tf)}",
    )
    if os.path.isdir(preferred):
        return preferred

    candidates = [
        d for d in os.listdir(analysis_dir)
        if d.startswith("origin_files_")
        and os.path.isdir(os.path.join(analysis_dir, d))
    ]
    if not candidates:
        return None

    candidates.sort()
    return os.path.join(analysis_dir, candidates[0])


def find_com_file(en_path):
    matches = glob(os.path.join(en_path, "data", "Com_All_Cells_Unwrapped_N_*.dat"))
    if not matches:
        return None
    matches.sort()
    return matches[0]


def read_positions_for_steps(filename, target_steps):
    target_steps = set(int(s) for s in target_steps)
    step_to_pos = {}
    ncell = None

    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(f.readline().strip())

            # ITEM: NUMBER OF CELLS
            _ = f.readline()
            ncell = int(f.readline().strip())

            # ITEM: BOX BOUNDS ...
            _ = f.readline()
            _ = f.readline()
            _ = f.readline()

            # ITEM: CELLS ...
            _ = f.readline()

            if step in target_steps:
                pos = np.empty((ncell, 2), dtype=float)
                for i in range(ncell):
                    parts = f.readline().split()
                    pos[i, 0] = float(parts[2])
                    pos[i, 1] = float(parts[3])
                step_to_pos[step] = pos
            else:
                for _ in range(ncell):
                    f.readline()

    return step_to_pos, ncell


def delta_r_for_origin(step_to_pos, origin_steps, upto):
    steps = [int(s) for s in origin_steps if int(s) in step_to_pos][:upto]
    if len(steps) < upto:
        raise ValueError(f"Only {len(steps)} origin steps found, need {upto}")

    ref = step_to_pos[steps[0]]
    delta_r = np.empty(upto, dtype=float)

    for i, step in enumerate(steps):
        disp = np.linalg.norm(step_to_pos[step] - ref, axis=1)
        delta_r[i] = disp.mean()

    time = (np.array(steps, dtype=float) - 1.0) * DT
    return time, delta_r


# ==========================================================
# Analysis Per v0
# ==========================================================

def analyze_v0_folder(v0_path, origin_steps):
    v0_value = float(os.path.basename(v0_path).split("v0_")[1])

    ensemble_dirs = [
        d for d in os.listdir(v0_path)
        if d.startswith("en") and os.path.isdir(os.path.join(v0_path, d))
    ]
    ensemble_dirs = numeric_sort(ensemble_dirs, "en")

    ensemble_curves = []
    flowing_count = 0
    error_ensembles = []
    time_axis = None

    target_steps = set()
    for steps in origin_steps:
        target_steps.update(steps[:UPTO])

    for en_folder in tqdm(ensemble_dirs, desc=f"{os.path.basename(v0_path)}"):
        en_path = os.path.join(v0_path, en_folder)
        try:
            com_file = find_com_file(en_path)
            if com_file is None:
                raise FileNotFoundError("Com_All_Cells_Unwrapped file missing")

            step_to_pos, ncell = read_positions_for_steps(com_file, target_steps)
            if ncell is None:
                raise ValueError("Could not read cell count from trajectory file")

            dr_total = np.zeros(UPTO, dtype=float)

            for i, one_origin_steps in enumerate(origin_steps):
                t, dr = delta_r_for_origin(step_to_pos, one_origin_steps, UPTO)
                dr_total += dr
                if i == 0:
                    time_axis = t

            dr_avg = dr_total / float(len(origin_steps))
            ensemble_curves.append(dr_avg)

            if np.max(dr_avg) >= np.sqrt(ncell):
                flowing_count += 1

        except Exception as e:
            error_ensembles.append(en_folder)
            print(f"Error in {os.path.basename(v0_path)}/{en_folder}: {e}")

    if ensemble_curves:
        ensemble_avg_dr = np.mean(np.vstack(ensemble_curves), axis=0)
        flowing_fraction = flowing_count / len(ensemble_curves)
        jammed_fraction = 1.0 - flowing_fraction
    else:
        ensemble_avg_dr = np.full(UPTO, np.nan, dtype=float)
        flowing_fraction = np.nan
        jammed_fraction = np.nan

    return {
        "v0": v0_value,
        "time": time_axis,
        "ensemble_avg_dr": ensemble_avg_dr,
        "flowing_fraction": flowing_fraction,
        "jammed_fraction": jammed_fraction,
        "num_ensembles": len(ensemble_curves),
        "error_ensembles": error_ensembles,
    }


# ==========================================================
# Analysis Per p0
# ==========================================================

def analyze_p0_folder(root_dir, p0_folder):
    print("\n========================================")
    print(f"Processing {p0_folder}")

    p0_value = float(p0_folder.split("p0_")[1])
    p0_path = os.path.join(root_dir, p0_folder)
    sim_path = os.path.join(p0_path, "T_0_Dr_0")
    search_path = sim_path if os.path.isdir(sim_path) else p0_path

    analysis_root = build_analysis_root(p0_path, search_path)
    output_dir = os.path.join(analysis_root, "absolute_distance_analysis")
    os.makedirs(output_dir, exist_ok=True)

    report_lines = [f"Analysis Report for {p0_folder}\n", f"Search path: {search_path}\n"]

    origin_dir = resolve_origin_dir(search_path, N_ORIGINS, ORIGIN_DIFF, TF)
    if origin_dir is None:
        report_lines.append("No origin_files_* directory found. Skipping.\n")
        with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
            f.writelines(report_lines)
        print(f"No origin files found for {p0_folder}")
        return

    report_lines.append(f"Origin dir: {origin_dir}\n")
    origin_steps = read_origin_steps(origin_dir, N_ORIGINS)

    v0_folders = [
        d for d in os.listdir(search_path)
        if d.startswith("v0_") and os.path.isdir(os.path.join(search_path, d))
    ]
    v0_folders = numeric_sort(v0_folders, "v0_")

    if not v0_folders:
        report_lines.append("No v0_* folders found. Skipping.\n")
        with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
            f.writelines(report_lines)
        print(f"No v0 folders found in {search_path}")
        return

    jammed_rows = []

    for v0_folder in tqdm(v0_folders, desc=f"{p0_folder}"):
        v0_path = os.path.join(search_path, v0_folder)
        result = analyze_v0_folder(v0_path, origin_steps)

        jammed_rows.append([result["v0"], result["jammed_fraction"]])

        if result["time"] is not None:
            np.savetxt(
                os.path.join(output_dir, f"Delta_r_en_avg_v0_{result['v0']}.txt"),
                np.c_[result["time"], result["ensemble_avg_dr"]],
                fmt=("%16.6f", "%16.6e"),
                header="time  Delta_r(t)",
            )

        report_lines.append(
            f"\nv0 = {result['v0']}\n"
            f"  Ensembles used: {result['num_ensembles']}\n"
            f"  Flowing fraction: {result['flowing_fraction']}\n"
            f"  Jammed fraction: {result['jammed_fraction']}\n"
            f"  Errors: {len(result['error_ensembles'])}\n"
            f"  Failed ensembles: {result['error_ensembles']}\n"
        )

    if jammed_rows:
        np.savetxt(
            os.path.join(output_dir, f"jammed_fraction_vs_v0_p0_{p0_value}.txt"),
            np.array(jammed_rows, dtype=float),
            fmt=("%10.4f", "%10.6f"),
            header="v0  jammed_fraction",
        )

    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.writelines(report_lines)

    print(f"Finished {p0_folder}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    p0_folders = [
        d for d in os.listdir(ROOT_DIR)
        if d.startswith("p0_") and os.path.isdir(os.path.join(ROOT_DIR, d))
    ]
    p0_folders = numeric_sort(p0_folders, "p0_")

    print("Detected p0 folders:", p0_folders)

    for p0_folder in p0_folders:
        analyze_p0_folder(ROOT_DIR, p0_folder)

    print("\nAll analysis completed successfully.")


if __name__ == "__main__":
    main()
