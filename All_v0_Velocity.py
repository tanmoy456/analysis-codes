import numpy as np
import os
from tqdm import tqdm


# ==========================================================
# Utility Functions
# ==========================================================

def numeric_sort(folder_list, prefix):
    """Sort folders like p0_3.9 numerically."""
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


def compute_histogram(data, n_bins=80):
    """Compute geometric-binned histogram."""
    if len(data) == 0:
        return None

    bins = np.geomspace(
        max(data.min(), 1e-6),
        data.max(),
        n_bins
    )

    hist, edges = np.histogram(data, bins=bins, density=False)
    centers = (edges[:-1] + edges[1:]) / 2

    return np.column_stack((centers, hist))


# ==========================================================
# Core Physics Function
# ==========================================================

def velocity_stats(filename, t_start):

    step_means = []
    step_numbers = []
    all_speeds = []

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

        # Last block
        if vx_list:
            vx = np.array(vx_list)
            vy = np.array(vy_list)
            speed = np.sqrt(vx**2 + vy**2)

            step_means.append(speed.mean())
            step_numbers.append(current_step)

            if current_step >= t_start:
                all_speeds.extend(speed)

    step_means = np.array(step_means)
    step_numbers = np.array(step_numbers)
    all_speeds = np.array(all_speeds)

    mask = step_numbers >= t_start
    steady_mean = step_means[mask].mean() if np.any(mask) else np.nan

    return steady_mean, all_speeds


# ==========================================================
# Analysis Per v0
# ==========================================================

def analyze_v0_folder(p0_path, v0_folder, t_start):

    v0_value = float(v0_folder.split("_")[1])
    v0_path = os.path.join(p0_path, v0_folder)

    ensemble_dirs = [
        d for d in os.listdir(v0_path)
        if d.startswith("en")
    ]

    ensemble_dirs = numeric_sort(ensemble_dirs, "en")

    ensemble_means = []
    global_distribution = []

    error_ensembles = []

    for en_folder in tqdm(ensemble_dirs, desc=f"{v0_folder}"):

        filename = os.path.join(
            v0_path,
            en_folder,
            "data",
            "velocity.dat"
        )

        try:
            if not os.path.exists(filename):
                raise FileNotFoundError("velocity.dat missing")

            mean_v, speeds = velocity_stats(filename, t_start)

            ensemble_means.append(mean_v)
            global_distribution.extend(speeds)

        except Exception as e:
            error_ensembles.append(en_folder)
            print(f"Error in {v0_folder}/{en_folder}")
            print(str(e))

    ensemble_means = np.array(ensemble_means)
    global_distribution = np.array(global_distribution)

    ensemble_avg = np.nanmean(ensemble_means) if len(ensemble_means) else np.nan

    return {
        "v0": v0_value,
        "ensemble_avg": ensemble_avg,
        "num_ensembles": len(ensemble_means),
        "error_ensembles": error_ensembles,
        "distribution": global_distribution
    }


# ==========================================================
# Analysis Per p0
# ==========================================================

def analyze_p0_folder(root_dir, p0_folder, t_start):

    print("\n========================================")
    print(f"Processing {p0_folder}")

    p0_value = float(p0_folder.split("_")[1])
    p0_path = os.path.join(root_dir, p0_folder)
    sim_path = os.path.join(p0_path, "T_0_Dr_0")

    # Support both structures:
    # 1) p0_x/T_0_Dr_0/v0_*
    # 2) p0_x/v0_* (legacy)
    search_path = sim_path if os.path.isdir(sim_path) else p0_path

    v0_folders = [
        d for d in os.listdir(search_path)
        if d.startswith("v0_") and os.path.isdir(os.path.join(search_path, d))
    ]

    v0_folders = numeric_sort(v0_folders, "v0_")

    analysis_root = build_analysis_root(p0_path, search_path)
    output_dir = os.path.join(analysis_root, "velocity_analysis")
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []

    report_lines = []
    report_lines.append(f"Analysis Report for {p0_folder}\n")
    report_lines.append(f"Search path: {search_path}\n")

    if not v0_folders:
        report_lines.append("No v0_* folders found. Skipping.\n")
        with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
            f.writelines(report_lines)
        print(f"No v0 folders found in {search_path}")
        return

    for v0_folder in v0_folders:

        result = analyze_v0_folder(search_path, v0_folder, t_start)

        summary_data.append([result["v0"], result["ensemble_avg"]])

        report_lines.append(
            f"\nv0 = {result['v0']}\n"
            f"  Ensembles used: {result['num_ensembles']}\n"
            f"  Errors: {len(result['error_ensembles'])}\n"
            f"  Failed ensembles: {result['error_ensembles']}\n"
        )

        # Save histogram
        hist_data = compute_histogram(result["distribution"])

        if hist_data is not None:
            np.savetxt(
                os.path.join(
                    output_dir,
                    f"velocity_distribution_v0_{result['v0']}.dat"
                ),
                hist_data,
                header="velocity  count",
                fmt="%.6e"
            )

    # Save summary file
    summary_data = np.array(summary_data)

    np.savetxt(
        os.path.join(
            output_dir,
            f"v0_vs_avg_velocity_p0_{p0_value}.dat"
        ),
        summary_data,
        header="v0  ensemble_avg_velocity",
        fmt="%.6e"
    )

    # Save analysis report
    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.writelines(report_lines)

    print("Finished", p0_folder)


# ==========================================================
# MAIN
# ==========================================================

def main():

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    t_start = 2e4

    p0_folders = [
        d for d in os.listdir(ROOT_DIR)
        if d.startswith("p0_") and os.path.isdir(os.path.join(ROOT_DIR, d))
    ]

    p0_folders = numeric_sort(p0_folders, "p0_")

    print("Detected p0 folders:", p0_folders)

    for p0_folder in p0_folders:
        analyze_p0_folder(ROOT_DIR, p0_folder, t_start)

    print("\nAll analysis completed successfully.")


if __name__ == "__main__":
    main()
