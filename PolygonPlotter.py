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

# --- LaTeX setup ---
from matplotlib import rc
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
rc('text', usetex=True)

# Threading control
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

class PolygonPlotter:
    def __init__(self, L, color_by='vertices', fill_color='colormap', 
                 show_com=False, show_cell_number=False, show_blue_box=True, 
                 fix_frame=True, plot_vertices_only=False, global_normalization=True,
                 show_tick_labels=True, show_axis_spines=True):
        """
        fill_color options:
        - 'colormap': Use colormap based on color_by property
        - 'white': All polygons filled white (black edges only)
        - 'black_vertices': Plot only black vertices, no polygons
        """
        self.L = L
        self.N = L * L
        self.hex_len = np.sqrt(2 / (3 * np.sqrt(3)))
        self.xm = L * np.sqrt(3) * self.hex_len
        self.ym = L / (np.sqrt(3) * self.hex_len)
        self.color_by = color_by
        self.fill_color = fill_color  # New parameter!
        self.show_com = show_com
        self.show_cell_number = show_cell_number
        self.show_blue_box = show_blue_box
        self.fix_frame = fix_frame
        self.plot_vertices_only = plot_vertices_only
        self.global_normalization = global_normalization
        self.show_tick_labels = show_tick_labels
        self.show_axis_spines = show_axis_spines

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
        if self.fill_color != 'colormap':  # No need for color bounds
            return 0, 1
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

    def _get_colorbar_label(self):
        if self.color_by == 'vertices':
            return r"$\mathrm{sides}$"
        elif self.color_by == 'area':
            return r"$A_i$"
        elif self.color_by == 'sigma_xy':
            return r"$\sigma_i$"
        elif self.color_by == 'pressure':
            return r"$pressure$"
        else:
            return self.color_by

    def plot_frame(self, args):
        (time, strain, vtx_data, prop_data, frame_number, bounds, color_by, 
         fill_color, show_com, show_cell_number, show_blue_box, fix_frame, 
         plot_vertices_only, output_dir, global_norm, show_tick_labels, show_axis_spines) = args

        fig, ax = plt.subplots(figsize=(8, 6))
        cells = {}
        
        # Build cell polygons
        for row in vtx_data:
            cell, x, y = int(row[0]), row[1], row[2]
            if cell not in cells:
                cells[cell] = []
            cells[cell].append((x, y))

        # Handle different fill_color modes
        if fill_color == 'colormap' and not plot_vertices_only:
            # Original colormap mode
            if global_norm:
                norm = mcolors.Normalize(vmin=bounds[0], vmax=bounds[1])
            else:
                values = [row[1 if color_by == 'vertices' else 
                             4 if color_by == 'area' else 
                             11 if color_by == 'sigma_xy' else (row[10] + row[12]) / 2] 
                         for row in prop_data]
                norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
            
            cmap = cm.viridis
            
            for row in prop_data:
                cell = int(row[0])
                if cell in cells:
                    color_value = row[1 if color_by == 'vertices' else 
                                    4 if color_by == 'area' else 
                                    11 if color_by == 'sigma_xy' else (row[10] + row[12]) / 2]
                    poly = Polygon(cells[cell], closed=True, edgecolor='black',
                                 facecolor=cmap(norm(color_value)), linewidth=0.3)
                    ax.add_patch(poly)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            label = self._get_colorbar_label()
            plt.colorbar(sm, label=label, ax=ax, shrink=0.8)
        
        elif fill_color == 'white' and not plot_vertices_only:
            # White fill with black edges
            for cell in cells:
                poly = Polygon(cells[cell], closed=True, 
                             facecolor='white', edgecolor='black', linewidth=0.5)
                ax.add_patch(poly)
        
        elif fill_color == 'black_vertices' or plot_vertices_only:
            # Only black vertices
            all_vertices = []
            for coords in cells.values():
                all_vertices.extend(coords)
            points = np.array(all_vertices)
            ax.scatter(points[:, 0], points[:, 1], c='black', s=2)

        # Blue box
        if show_blue_box:
            x1, x2 = 0, bounds[2]
            y1, y2 = 0, bounds[3]
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b-', linewidth=1)

        # Frame limits
        if fix_frame:
            ax.set_xlim(-2, bounds[2] + 2)
            ax.set_ylim(-2, bounds[3] + 2)

        # Title
        plt.title(rf'$\gamma = {strain:.6f}$', fontsize=10)

        # Tick labels and spines
        if not show_tick_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)
        if not show_axis_spines:
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.savefig(f"{output_dir}/snapshot_frame_{frame_number:05d}.png", 
                   bbox_inches="tight", pad_inches=0.01, dpi=200, facecolor='white')
        plt.close(fig)

    def generate_plots(self, output_dir, num_processes=8, strain_range=None):
        os.makedirs(output_dir, exist_ok=True)
        bounds = (self.global_min, self.global_max, self.xm, self.ym)
        all_keys = sorted(self.vertex_time_series_data.keys())
        selected_keys = all_keys if strain_range is None else \
                       [k for k in all_keys if strain_range[0] <= k[1] <= strain_range[1]]

        if not selected_keys:
            raise ValueError(f"No frames found in strain range {strain_range}!")

        args_list = []
        for i, key in enumerate(selected_keys):
            args_list.append((key[0], key[1], self.vertex_time_series_data[key], 
                            self.property_time_series_data.get(key, []), i+1, bounds,
                            self.color_by, self.fill_color, self.show_com, self.show_cell_number,
                            self.show_blue_box, self.fix_frame, self.plot_vertices_only,
                            output_dir, self.global_normalization, self.show_tick_labels,
                            self.show_axis_spines))
        
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            list(tqdm(pool.map(self.plot_frame, args_list), total=len(args_list), 
                     desc="Generating frames"))

# Usage examples:
gd = '0.0001'
en = 'en1'
output_dir = f"config_figures_gd_{gd}_en_{en}_white_fill"
L = 10

# Option 1: Original colormap (default)
#plotter1 = PolygonPlotter(L, color_by='sigma_xy', fill_color='colormap')
#plotter1.parse_time_series_data(f'../gd_{gd}/{en}/data/VertexPositions_N_{L*L}.dat',
#                                f'../gd_{gd}/{en}/data/Cell_propery_N_{L*L}.dat')
#plotter1.generate_plots(output_dir, num_processes=4)

# Option 2: White fill polygons
#output_dir_white = f"config_figures_gd_{gd}_en_{en}_white_fill"
#plotter2 = PolygonPlotter(L, fill_color='white')
#plotter2.parse_time_series_data(f'../gd_{gd}/{en}/data/VertexPositions_N_{L*L}.dat',
#                                f'../gd_{gd}/{en}/data/Cell_propery_N_{L*L}.dat')
#plotter2.generate_plots(output_dir_white, num_processes=4)

# Option 3: Black vertices only
output_dir_vertices = f"config_figures_gd_{gd}_en_{en}_black_vertices"
plotter3 = PolygonPlotter(L, fill_color='black_vertices')
plotter3.parse_time_series_data(f'../gd_{gd}/{en}/data/VertexPositions_N_{L*L}.dat',
                                f'../gd_{gd}/{en}/data/Cell_propery_N_{L*L}.dat')
plotter3.generate_plots(output_dir_vertices, num_processes=4)

