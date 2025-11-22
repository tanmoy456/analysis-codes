import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

class PolygonPlotter:
    def __init__(self, L, color_by='vertices', show_com=False, show_cell_number=False,
                 show_blue_box=True, fix_frame=True, plot_vertices_only=False,
                 global_normalization=True, show_tick_labels=True, show_axis_spines=True):  # added show_tick_labels
        self.L = L
        self.N = L * L
        self.hex_len = np.sqrt(2 / (3 * np.sqrt(3)))
        self.xm = L * np.sqrt(3) * self.hex_len
        self.ym = L / (np.sqrt(3) * self.hex_len)
        self.color_by = color_by
        self.show_com = show_com
        self.show_cell_number = show_cell_number
        self.show_blue_box = show_blue_box
        self.fix_frame = fix_frame
        self.plot_vertices_only = plot_vertices_only
        self.global_normalization = global_normalization
        self.show_tick_labels = show_tick_labels
        self.show_axis_spines = show_axis_spines  # Controls whether to show black border

    def parse_time_series_data(self, vertex_filename, property_filename):
        self.vertex_time_series_data = self._parse_file(vertex_filename)
        self.property_time_series_data = self._parse_file(property_filename)
        self.global_min, self.global_max = self._compute_global_bounds()

    def _parse_file(self, filename):
        data = {}
        with open(filename, "r") as f:
            lines = f.readlines()
        current_time, current_strain = None, None
        for line in lines:
            line = line.strip()
            if line.startswith('Time'):
                current_time = float(line.split(":")[1].strip())
            elif line.startswith('Strain'):
                current_strain = float(line.split(":")[1].strip())
                data[(current_time, current_strain)] = []
            elif line and current_time is not None:
                row = [float(x) if i > 0 else int(x) for i, x in enumerate(line.split())]
                data[(current_time, current_strain)].append(row)
        return data

    def _compute_global_bounds(self):
        all_vals = []
        for time_strain, data in self.property_time_series_data.items():
            for row in data:
                if self.color_by == 'vertices':
                    all_vals.append(row[1])
                elif self.color_by == 'area':
                    all_vals.append(row[4])
                elif self.color_by == 'sigma_xy':
                    all_vals.append(row[11])
                elif self.color_by == 'pressure':
                    all_vals.append((row[10] + row[12]) / 2)
        return min(all_vals), max(all_vals)

    def plot_frame(self, args):
        time, strain, vtx_data, prop_data, frame_number, bounds, color_by, show_com, show_cell_number, show_blue_box, fix_frame, plot_vertices_only, output_dir, global_norm, show_tick_labels = args
        fig, ax = plt.subplots(figsize=(8, 6))
        cells = {}
        for row in vtx_data:
            cell, x, y = int(row[0]), row[1], row[2]
            if cell not in cells:
                cells[cell] = []
            cells[cell].append((x, y))

        # Determine normalization range
        if global_norm:
            norm = mcolors.Normalize(vmin=bounds[0], vmax=bounds[1])
        else:
            values = []
            for row in prop_data:
                if color_by == 'vertices':
                    values.append(row[1])
                elif color_by == 'area':
                    values.append(row[4])
                elif color_by == 'sigma_xy':
                    values.append(row[11])
                elif color_by == 'pressure':
                    values.append((row[10] + row[12]) / 2)
            if values:
                local_min = min(values)
                local_max = max(values)
            else:
                local_min, local_max = 0, 1  # fallback range
            norm = mcolors.Normalize(vmin=local_min, vmax=local_max)

        cmap = cm.viridis

        if plot_vertices_only:
            for cell, coords in cells.items():
                points = np.array(coords)
                ax.scatter(points[:, 0], points[:, 1], s=1)
        else:
            for row in prop_data:
                cell = int(row[0])
                if cell in cells:
                    if color_by == 'vertices':
                        color_value = row[1]
                    elif color_by == 'area':
                        color_value = row[4]
                    elif color_by == 'sigma_xy':
                        color_value = row[11]
                    elif color_by == 'pressure':
                        color_value = (row[10] + row[12]) / 2
                    else:
                        color_value = 0
                    poly = Polygon(cells[cell], closed=True, edgecolor='black',
                                   facecolor=cmap(norm(color_value)), linewidth=0.3)
                    ax.add_patch(poly)

        if fix_frame:
            ax.set_xlim(-2, bounds[2] + 2)
            ax.set_ylim(-2, bounds[3] + 2)
        if show_blue_box:
            x1, x2 = 0, bounds[2]
            y1, y2 = 0, bounds[3]
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b-', linewidth=1)
        if not plot_vertices_only:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, label=color_by.capitalize(), ax=ax, shrink=0.8)
        plt.title(rf'$\gamma = {strain:.6f}$', fontsize=10)

        # Hide or show tick labels
        if not show_tick_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # Also set tick size (major and minor) to zero to remove vertical/horizontal small ticks
            ax.tick_params(axis='both', which='both', length=0)

        if not self.show_axis_spines:
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.savefig(f"{output_dir}/snapshot_frame_{frame_number:05d}.png", bbox_inches="tight",
                    pad_inches=0.01, dpi=200, facecolor='white')
        plt.close(fig)

    def generate_plots(self, output_dir, num_processes=8):
        os.makedirs(output_dir, exist_ok=True)
        bounds = (self.global_min, self.global_max, self.xm, self.ym)
        args_list = []
        for i, (time_strain) in enumerate(self.vertex_time_series_data.keys()):
            vtx_data = self.vertex_time_series_data[time_strain]
            prop_data = self.property_time_series_data.get(time_strain, [])
            args_list.append((time_strain[0], time_strain[1], vtx_data, prop_data, i+1, bounds,
                              self.color_by, self.show_com, self.show_cell_number, self.show_blue_box,
                              self.fix_frame, self.plot_vertices_only, output_dir, self.global_normalization, self.show_tick_labels))
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            list(tqdm(pool.map(self.plot_frame, args_list), total=len(args_list), desc="Generating frames"))

# Usage example
gd = '0.0001'
en = 'en1'
output_dir = f"config_figures_gd_{gd}_en_{en}_local_norm_color"
os.makedirs(output_dir, exist_ok=True)
L = 10
N = L * L
plotter = PolygonPlotter(L, color_by='sigma_xy',
                         show_com=False,
                         show_cell_number=False,
                         show_blue_box=True,
                         fix_frame=True,
                         global_normalization=False,
                         show_tick_labels=False,   # Hide tick labels
                         show_axis_spines=False)   # Hide black border

plotter.parse_time_series_data(f'../gd_{gd}/{en}/data/VertexPositions_N_{N}.dat',
                               f'../gd_{gd}/{en}/data/Cell_propery_N_{N}.dat')
plotter.generate_plots(output_dir, num_processes=4)
