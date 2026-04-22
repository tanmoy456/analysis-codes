import numpy as np
import matplotlib.pyplot as plt
import os
import math


class T1Analyzer:

    def __init__(self,
                 gd='0.0001',
                 ensembles=None,
                 gamma_cut=0.1,
                 gamma_max=None,
                 gamma_window=None,
                 L=30,
                 factor=1.2,
                 normalize=False,
                 figsize=(8, 6),
                 dpi=300,
                 plot_scatter=True,
                 plot_activity=True,
                 plot_profile=False,
                 n_per_fig=16,

                 # NEW
                 n_time_bins=100,
                 accumulate=True,
                 delta_gamma=None):

        self.gd = gd
        self.ensembles = ensembles
        self.gamma_cut = gamma_cut
        self.gamma_max = gamma_max
        self.gamma_window = gamma_window
        self.normalize = normalize

        self.figsize = figsize
        self.dpi = dpi

        self.plot_scatter = plot_scatter
        self.plot_activity = plot_activity
        self.plot_profile = plot_profile

        self.n_per_fig = n_per_fig

        # NEW
        self.n_time_bins = n_time_bins
        self.accumulate = accumulate
        self.delta_gamma = delta_gamma

        # =========================
        # SYSTEM SIZE
        # =========================
        hex_len = np.sqrt(2 / (3 * np.sqrt(3)))
        self.Lx = L * np.sqrt(3) * hex_len
        self.Ly = L / (np.sqrt(3) * hex_len)

        self.Nx = int(self.Lx * factor)
        self.Ny = int(self.Ly * factor)

        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny

        print("Lx, Ly =", self.Lx, self.Ly)
        print("Nx, Ny =", self.Nx, self.Ny)
        print("dx, dy =", self.dx, self.dy)

        self.base_dir = f"plots_gd_{gd}"
        os.makedirs(self.base_dir, exist_ok=True)

        self.activities = []

    # =========================
    def load_data(self, en):
        data = np.loadtxt(f"../gd_{self.gd}/{en}/data/T1_field.dat")
        return data[:, 0], data[:, 1], data[:, 2]

    # =========================
    def compute_activity(self, x, y):
        activity = np.zeros((self.Nx, self.Ny))

        i = (x / self.dx).astype(int)
        j = (y / self.dy).astype(int)

        i = np.clip(i, 0, self.Nx - 1)
        j = np.clip(j, 0, self.Ny - 1)

        np.add.at(activity, (i, j), 1)
        return activity

    # =========================
    def filter_data(self, gamma, x, y):

        mask = gamma >= self.gamma_cut
        if self.gamma_max is not None:
            mask &= (gamma <= self.gamma_max)

        gamma = gamma[mask]
        x = x[mask]
        y = y[mask]

        # boundary safety
        valid = (x >= 0) & (x <= self.Lx) & (y >= 0) & (y <= self.Ly)
        return gamma[valid], x[valid], y[valid]

    # =========================
    def plot_activity_time_series(self, en, gamma, x, y):

        en_dir = os.path.join(self.base_dir, en, "timeseries")
        os.makedirs(en_dir, exist_ok=True)

        # sort by gamma
        idx = np.argsort(gamma)
        gamma = gamma[idx]
        x = x[idx]
        y = y[idx]

        gmin = gamma.min()
        gmax = gamma.max()

        # ---- binning strategy
        if self.delta_gamma is not None:
            bins = np.arange(gmin, gmax + self.delta_gamma, self.delta_gamma)
            n_bins = len(bins) - 1
        else:
            n_bins = self.n_time_bins
            bins = np.linspace(gmin, gmax, n_bins + 1)

        print(f"[{en}] frames={n_bins}, Δγ={(gmax-gmin)/n_bins:.4f}")

        activity_accum = np.zeros((self.Nx, self.Ny))

        # ---- global color scale (important for video consistency)
        vmax = 0

        temp_fields = []

        for k in range(n_bins):

            g0 = bins[k]
            g1 = bins[k+1]

            mask_bin = (gamma >= g0) & (gamma < g1)

            if not np.any(mask_bin):
                temp_fields.append(activity_accum.copy())
                continue

            activity_bin = self.compute_activity(x[mask_bin], y[mask_bin])

            if self.accumulate:
                activity_accum += activity_bin
                field = activity_accum.copy()
            else:
                field = activity_bin

            temp_fields.append(field)
            vmax = max(vmax, field.max())

        # ---- plotting
        for k, field in enumerate(temp_fields):

            plt.figure(figsize=self.figsize)
            plt.imshow(field.T,
                       origin='lower',
                       extent=[0, self.Lx, 0, self.Ly],
                       vmin=0, vmax=vmax)

            plt.colorbar(label="Accumulated T1 count")
            plt.title(f"{en} frame {k}")

            plt.savefig(f"{en_dir}/frame_{k:04d}.png", dpi=self.dpi)
            plt.close()

    # =========================
    def process_ensemble(self, en):

        print(f"\nProcessing {en}")

        gamma, x, y = self.load_data(en)

        gamma, x, y = self.filter_data(gamma, x, y)

        # ---- FULL activity (optional)
        if self.plot_activity:
            activity = self.compute_activity(x, y)

            en_dir = os.path.join(self.base_dir, en)
            os.makedirs(en_dir, exist_ok=True)

            plt.figure(figsize=self.figsize)
            plt.imshow(activity.T, origin='lower',
                       extent=[0, self.Lx, 0, self.Ly])
            plt.colorbar(label="T1 count")
            plt.title(f"{en} activity (full)")
            plt.savefig(f"{en_dir}/activity_full.png", dpi=self.dpi)
            plt.close()

        # ---- TIME SERIES (NEW CORE FEATURE)
        self.plot_activity_time_series(en, gamma, x, y)

    # =========================
    def run(self):

        for en in self.ensembles:
            self.process_ensemble(en)


# =========================
# USAGE
# =========================
if __name__ == "__main__":

    ensembles = [f'en{idx}' for idx in range(1, 9)]

    analyzer = T1Analyzer(
        L=20,
        gd='0.0001',
        ensembles=ensembles,
        gamma_cut=4.0,
        gamma_max=12.0,

        # KEY SETTINGS
        n_time_bins=200,     # number of frames
        accumulate=True,     # cumulative activity

        # OR use fixed Δγ instead:
        # delta_gamma=0.1,

        factor=1.2,
        plot_activity=True
    )

    analyzer.run()
