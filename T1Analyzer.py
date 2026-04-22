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
                 n_per_fig=16):

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

        print("gamma_cut =", self.gamma_cut)
        print("gamma_max =", self.gamma_max)
        print("gamma_window =", self.gamma_window)

        # =========================
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
    def process_ensemble(self, en):

        print(f"\nProcessing {en}")

        en_dir = os.path.join(self.base_dir, en)
        os.makedirs(en_dir, exist_ok=True)

        gamma, x, y = self.load_data(en)

        print("gamma min =", gamma.min(), "gamma max =", gamma.max())

        # ---- apply filters
        mask = gamma >= self.gamma_cut
        if self.gamma_max is not None:
            mask &= (gamma <= self.gamma_max)

        gamma_ss = gamma[mask]
        x_ss = x[mask]
        y_ss = y[mask]

        # =========================
        # SCATTER FULL
        # =========================
        if self.plot_scatter:
            plt.figure(figsize=self.figsize)
            plt.scatter(x_ss, y_ss, s=5)
            plt.xlim(0, self.Lx)
            plt.ylim(0, self.Ly)
            plt.title(f"{en} scatter (full)")
            plt.savefig(f"{en_dir}/scatter_full.png", dpi=self.dpi)
            plt.close()

        # =========================
        # ACTIVITY FULL
        # =========================
        activity = self.compute_activity(x_ss, y_ss)

        if self.plot_activity:
            plt.figure(figsize=self.figsize)
            plt.imshow(activity.T, origin='lower',
                       extent=[0, self.Lx, 0, self.Ly])
            plt.colorbar(label="T1 count")
            plt.title(f"{en} activity (full)")
            plt.savefig(f"{en_dir}/activity_full.png", dpi=self.dpi)
            plt.close()

        self.activities.append(activity)

        # =========================
        # WINDOWED SNAPSHOTS
        # =========================
        if self.gamma_window is not None:

            g0 = self.gamma_cut
            gmax = self.gamma_max if self.gamma_max else gamma_ss.max()

            k = 0
            while g0 < gmax:

                g1 = min(g0 + self.gamma_window, gmax)

                wmask = (gamma >= g0) & (gamma < g1)

                x_w = x[wmask]
                y_w = y[wmask]

                if len(x_w) == 0:
                    g0 = g1
                    continue

                # ---- scatter window
                if self.plot_scatter:
                    plt.figure(figsize=self.figsize)
                    plt.scatter(x_w, y_w, s=5)
                    plt.xlim(0, self.Lx)
                    plt.ylim(0, self.Ly)
                    plt.title(f"{en}: γ [{g0:.2f}, {g1:.2f}]")
                    plt.savefig(f"{en_dir}/scatter_window_{k}.png",
                                dpi=self.dpi)
                    plt.close()

                # ---- activity window
                activity_w = self.compute_activity(x_w, y_w)

                if self.plot_activity:
                    plt.figure(figsize=self.figsize)
                    plt.imshow(activity_w.T, origin='lower',
                               extent=[0, self.Lx, 0, self.Ly])
                    plt.colorbar(label="T1 count")
                    plt.title(f"{en}: γ [{g0:.2f}, {g1:.2f}]")
                    plt.savefig(f"{en_dir}/activity_window_{k}.png",
                                dpi=self.dpi)
                    plt.close()

                k += 1
                g0 = g1

    # =========================
    def plot_all_activity_grids(self):

        agg_dir = os.path.join(self.base_dir, "ALL")
        os.makedirs(agg_dir, exist_ok=True)

        n_total = len(self.activities)
        n_per_fig = self.n_per_fig

        n_figs = math.ceil(n_total / n_per_fig)

        for f in range(n_figs):

            start = f * n_per_fig
            end = min((f + 1) * n_per_fig, n_total)

            subset = self.activities[start:end]
            labels = self.ensembles[start:end]

            ncols = int(np.ceil(np.sqrt(n_per_fig)))
            nrows = int(np.ceil(len(subset) / ncols))

            fig, axes = plt.subplots(nrows, ncols,
                                    figsize=(3*ncols, 3*nrows),
                                    constrained_layout=True)

            axes = np.array(axes).reshape(-1)

            for i, ax in enumerate(axes):

                if i < len(subset):

                    im = ax.imshow(subset[i].T,
                                   origin='lower',
                                   extent=[0, self.Lx, 0, self.Ly])

                    ax.set_title(labels[i], fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    fig.colorbar(im, ax=ax,
                                 fraction=0.046,
                                 pad=0.04)
                else:
                    ax.axis('off')

            plt.savefig(f"{agg_dir}/activity_grid_{f}.png",
                        dpi=self.dpi)
            plt.close()

    # =========================
    def run(self):

        for en in self.ensembles:
            self.process_ensemble(en)

        self.plot_all_activity_grids()


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
        gamma_window=2.0,
        factor=1.2,
        plot_scatter=True,
        plot_activity=True,
        n_per_fig=8
    )

    analyzer.run()
