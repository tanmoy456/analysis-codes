import os
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tqdm import tqdm
import concurrent.futures
import gc


## --- LaTeX setup ---
from matplotlib import rc
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'
rc('text', usetex=True)

class AspectRatioPlotter:
    def __init__(self, L, output_dir='aspect_ratio_figures', show_blue_box=True, 
                 fix_frame=True, show_cell_number=False, num_processes=1,
                 show_tick_labels=True, show_axis_spines=True, show_title=True, title_mode='strain',
                 global_normalization=True):
        # Constants for plot limits
        self.L = L
        self.N = L * L
        self.hex_len = np.sqrt(2 / (3 * np.sqrt(3)))
        self.xm = L * np.sqrt(3) * self.hex_len
        self.ym = L / (np.sqrt(3) * self.hex_len)

        # Set options for plotting
        self.output_dir = output_dir
        self.show_cell_number = show_cell_number
        self.show_blue_box = show_blue_box
        self.fix_frame = fix_frame
        self.show_tick_labels = show_tick_labels
        self.show_axis_spines = show_axis_spines
        self.show_title = show_title
        self.title_mode = title_mode
        self.global_normalization = global_normalization
        
        # Data containers
        self.vertex_time_series_data = {}
        self.aspect_ratios = {}
        self.global_min = None
        self.global_max = None
        
        # Number of processes for parallel execution
        self.num_processes = num_processes

    def parse_time_series_data(self, filename):
        """Parses vertex time series data from the file."""
        with open(filename, 'r') as file:
            data = file.readlines()

        time_series_data = {}
        current_time = None
        current_strain = None

        for line in data:
            line = line.strip()
            if line.startswith('Time'):
                try:
                    current_time = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    print(f"Warning: Unable to parse time from line: {line}")
                    continue
            elif line.startswith('Strain'):
                try:
                    current_strain = float(line.split(":")[1].strip())
                    time_series_data[(current_time, current_strain)] = []
                except (IndexError, ValueError):
                    print(f"Warning: Unable to parse strain from line: {line}")
                    continue
            elif line and current_time is not None:
                try:
                    row = [float(x) if i > 0 else int(x) for i, x in enumerate(line.split())]
                    time_series_data[(current_time, current_strain)].append(row)
                except ValueError:
                    print(f"Warning: Unable to parse data from line: {line}")
                    continue

        self.vertex_time_series_data = time_series_data
        print(f"Loaded {len(time_series_data)} time steps")

    # ... [polygon_centroid, polygon_inertia_centroidal, compute_aspect_ratio, calculate_all_aspect_ratios methods unchanged] ...
    
    def polygon_centroid(self, vertices):
        """Calculate the centroid of a polygon using the shoelace formula."""
        x = vertices[:, 0]
        y = vertices[:, 1]
        
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        
        cross = x * y_next - x_next * y
        A = 0.5 * np.sum(cross)
        
        if abs(A) < 1e-10:
            return np.mean(x), np.mean(y), 0.0
        
        cx = (1/(6*A)) * np.sum((x + x_next) * cross)
        cy = (1/(6*A)) * np.sum((y + y_next) * cross)
        
        return cx, cy, abs(A)
    
    def polygon_inertia_centroidal(self, vertices):
        """Calculate centroidal moment of inertia using the polygon formula."""
        vertices = np.array(vertices)
        
        # Calculate centroid and area
        cx, cy, area = self.polygon_centroid(vertices)
        
        if area < 1e-10:
            return 0.0, 0.0, 0.0, area
        
        # Shift to centroid frame
        x_cent = vertices[:, 0] - cx
        y_cent = vertices[:, 1] - cy
        
        # Cyclic indexing for closed polygon
        x_cent_next = np.roll(x_cent, -1)
        y_cent_next = np.roll(y_cent, -1)
        
        cross_cent = x_cent * y_cent_next - x_cent_next * y_cent
        
        # Calculate moments of inertia
        Iy = (1/12) * np.sum(cross_cent * (x_cent**2 + x_cent*x_cent_next + x_cent_next**2))
        Ix = (1/12) * np.sum(cross_cent * (y_cent**2 + y_cent*y_cent_next + y_cent_next**2))
        Ixy = (1/24) * np.sum(cross_cent * (x_cent*y_cent_next + 2*x_cent*y_cent + 
                                             2*x_cent_next*y_cent_next + x_cent_next*y_cent))
        
        return Ix, Iy, Ixy, area
    
    def compute_aspect_ratio(self, Ix, Iy, Ixy):
        """Compute aspect ratio from inertia tensor eigenvalues."""
        inertia_tensor = np.array([[Ix, Ixy], [Ixy, Iy]])
        
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        eigenvalues = np.abs(eigenvalues)
        
        if eigenvalues[0] < 1e-10:
            return 1.0
        
        lambda_min, lambda_max = np.sort(eigenvalues)
        aspect_ratio = math.sqrt(lambda_max / lambda_min)
        
        return aspect_ratio
    
    def calculate_all_aspect_ratios(self, strain_range=None, time_range=None):
        """Calculate aspect ratios only for selected time/strain range."""
        print("\nCalculating aspect ratios for selected time steps...")

        all_keys = sorted(self.vertex_time_series_data.keys())  # (time, strain)

        if strain_range is not None and time_range is not None:
            raise ValueError("Use either strain_range or time_range, not both.")
        elif strain_range is not None:
            selected_keys = [k for k in all_keys if strain_range[0] <= k[1] <= strain_range[1]]
        elif time_range is not None:
            selected_keys = [k for k in all_keys if time_range[0] <= k[0] <= time_range[1]]
        else:
            selected_keys = all_keys

        if not selected_keys:
            raise ValueError(f"No frames found for given range: "
                             f"strain_range={strain_range}, time_range={time_range}")

        for key in tqdm(selected_keys, desc="Processing time steps"):
            time, strain = key
            vertex_data = self.vertex_time_series_data[key]

            # Organize vertices by cell
            cells = {}
            for row in vertex_data:
                cell = int(row[0])
                x, y = row[1], row[2]
                if cell not in cells:
                    cells[cell] = []
                cells[cell].append([x, y])

            time_step_aspect_ratios = {}
            for cell, vertices in cells.items():
                vertices = np.array(vertices)

                if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
                    vertices = vertices[:-1]
                if len(vertices) < 3:
                    continue

                Ix, Iy, Ixy, area = self.polygon_inertia_centroidal(vertices)
                if area < 1e-10:
                    continue

                aspect_ratio = self.compute_aspect_ratio(Ix, Iy, Ixy)
                time_step_aspect_ratios[cell] = aspect_ratio

            self.aspect_ratios[key] = time_step_aspect_ratios


    def compute_global_min_max(self):
        """Computes global min and max aspect ratio values across all time steps."""
        all_values = []
        for cell_aspect_ratios in self.aspect_ratios.values():
            all_values.extend(cell_aspect_ratios.values())
        
        self.global_min = min(all_values)
        self.global_max = max(all_values)
        print(f"Global aspect ratio range: [{self.global_min:.3f}, {self.global_max:.3f}]")

    def generate_plots(self, strain_range=None, time_range=None):
        """
        Generates and saves the plots for specified time/strain range.
        Use either strain_range or time_range (not both).
        """
        self.compute_global_min_max()
        
        # Filter frames by range
        all_keys = sorted(self.vertex_time_series_data.keys())
        if strain_range is not None and time_range is not None:
            raise ValueError("Use either strain_range or time_range, not both.")
        elif strain_range is not None:
            selected_keys = [k for k in all_keys if strain_range[0] <= k[1] <= strain_range[1]]
        elif time_range is not None:
            selected_keys = [k for k in all_keys if time_range[0] <= k[0] <= time_range[1]]
        else:
            selected_keys = all_keys

        if not selected_keys:
            raise ValueError(f"No frames found for given range: strain_range={strain_range}, time_range={time_range}")

        print(f"Generating {len(selected_keys)} frames")
        
        # Prepare all plot parameters upfront
        plot_params_list = []
        for frame_number, key in enumerate(selected_keys, start=1):
            time, strain = key
            vertex_data = self.vertex_time_series_data[key]
            aspect_ratio_data = self.aspect_ratios.get(key, {})
            
            plot_params = {
                'time': time,
                'strain': strain,
                'frame_number': frame_number,
                'vertex_data': vertex_data,
                'aspect_ratio_data': aspect_ratio_data,
                'L': self.L,
                'xm': self.xm,
                'ym': self.ym,
                'show_cell_number': self.show_cell_number,
                'show_blue_box': self.show_blue_box,
                'fix_frame': self.fix_frame,
                'show_tick_labels': self.show_tick_labels,
                'show_axis_spines': self.show_axis_spines,
                'show_title': self.show_title,
                'title_mode': self.title_mode,
                'global_min': self.global_min,
                'global_max': self.global_max,
                'global_normalization': self.global_normalization,
                'output_dir': self.output_dir
            }
            plot_params_list.append(plot_params)
        
        # Use ProcessPoolExecutor with standalone function
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            list(tqdm(
                executor.map(plot_single_frame_aspect, plot_params_list),
                total=len(plot_params_list),
                desc="Generating plots"
            ))


# Updated standalone function for multiprocessing (must be at module level)
def plot_single_frame_aspect(params):
    """Standalone function to plot a single frame colored by aspect ratio."""
    fig = None
    try:
        # Extract parameters
        time = params['time']
        strain = params['strain']
        frame_number = params['frame_number']
        vertex_data = params['vertex_data']
        aspect_ratio_data = params['aspect_ratio_data']
        xm = params['xm']
        ym = params['ym']
        show_cell_number = params['show_cell_number']
        show_blue_box = params['show_blue_box']
        fix_frame = params['fix_frame']
        show_tick_labels = params['show_tick_labels']
        show_axis_spines = params['show_axis_spines']
        show_title = params['show_title']
        title_mode = params['title_mode']
        global_min = params['global_min']
        global_max = params['global_max']
        global_normalization = params['global_normalization']
        output_dir = params['output_dir']

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Organize vertices by cell
        cells = {}
        for row in vertex_data:
            cell, x, y = int(row[0]), row[1], row[2]
            if cell not in cells:
                cells[cell] = {'x': [], 'y': []}
            cells[cell]['x'].append(x)
            cells[cell]['y'].append(y)

        # Set up color mapping (respect global_normalization)
        if global_normalization:
            norm = mcolors.Normalize(vmin=global_min, vmax=global_max)
        else:
            # Frame-local normalization
            valid_aspects = [aspect_ratio_data.get(cell, np.nan) for cell in cells.keys()]
            valid_aspects = [v for v in valid_aspects if not np.isnan(v)]
            if valid_aspects:
                norm = mcolors.Normalize(vmin=min(valid_aspects), vmax=max(valid_aspects))
            else:
                norm = mcolors.Normalize(vmin=global_min, vmax=global_max)

        cmap = cm.viridis

        # Plot each cell
        for cell, coord_data in cells.items():
            coords = list(zip(coord_data['x'], coord_data['y']))
            
            # Get aspect ratio for this cell
            aspect_ratio = aspect_ratio_data.get(cell, np.nan)
            
            if not np.isnan(aspect_ratio):
                color = cmap(norm(aspect_ratio))
            else:
                color = 'gray'  # For cells without aspect ratio
            
            polygon = Polygon(coords, closed=True, edgecolor='black', 
                            facecolor=color, linewidth=0.5)
            ax.add_patch(polygon)

            if show_cell_number:
                com_x = np.mean(coord_data['x'])
                com_y = np.mean(coord_data['y'])
                ax.text(com_x, com_y, str(cell), ha='center', va='center', 
                       fontsize=8, color='white', weight='bold')

        # Set axis limits
        if fix_frame:
            ax.set_xlim(0-1.9, xm+1.9)
            ax.set_ylim(0-1.4, ym+1.2)
        
        ax.set_aspect('equal')

        # Draw bounding box
        if show_blue_box:
            x1, x2 = 0, xm
            y1, y2 = 0, ym
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b-', linewidth=2)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax,  #label='Aspect Ratio', 
                         orientation='horizontal',  # colorbar below instead of side
                         shrink=0.62,      # Shortens LENGTH (vertical size for vertical cbar)
                         aspect=18,        # Controls WIDTH (higher = thinner)
                         fraction=0.05,    # Overall cbar size relative to axes
                         pad=0.01,          # GAP between plot and cbar
                    )

        ##**FULL TEXT/LABEL CONTROL**
        cbar.set_label(r'$\mathrm{Aspect\ Ratio}$',
                       fontsize=24,      # Text size
                       labelpad=12,      # Distance from cbar to text
                       # rotation=270,    # 270° = vertical (default for vertical cbar)
                       ha='center',      # Horizontal alignment
                       va='center'       # Vertical alignment
                    )
        # ** FULL TICK CONTROL **
        cbar.ax.tick_params(                   # Access cbar's axes for ticks
                        direction='in',        # 'in' = inside | 'out' = outside (default)
                        labelsize=24,          # Tick label font size
                        size=15,               # Tick mark length
                        width=1.0,             # Tick mark thickness
                        pad=1,                 # Space between tick & label
                        # which='major'        # Only major ticks
                        bottom=True, top=False,
                    )

        # Title based on title_mode
        if show_title:
            if title_mode == 'strain':
                ax.set_title(rf'$\gamma = {strain:.3f}$', fontsize=30)
            elif title_mode == 'time':
                ax.set_title(rf'$t = {time:.3f}$', fontsize=30)

        # Control tick labels and spines
        if not show_tick_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)
        if not show_axis_spines:
            for spine in ax.spines.values():
                spine.set_visible(False)

        # Save to the correct output directory
        output_path = os.path.join(output_dir, f"aspect_ratio_frame_{frame_number:05d}.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.01, facecolor='white')

    except Exception as e:
        print(f"Error plotting frame {params.get('frame_number', '?')}: {e}")
    
    finally:
        # Always close the figure and force garbage collection
        if fig is not None:
            plt.close(fig)
        plt.close('all')
        gc.collect()


# Updated usage example
if __name__ == '__main__':
    prefix = 'gd'
    prefix_val = '0.00001'
    en = 'en3'
    output_dir = f"aspect_ratio_snapshots_{prefix}_{prefix_val}_{en}"
    os.makedirs(output_dir, exist_ok=True)

    L = 10
    N = L * L
    
    plotter = AspectRatioPlotter(
        L=L, 
        output_dir=output_dir,
        show_cell_number=False,  
        show_blue_box=True, 
        fix_frame=True,
        show_tick_labels=False,      # NEW: Hide tick labels
        show_axis_spines=False,      # NEW: Hide axis spines  
        show_title=True,            # NEW: Hide title
        title_mode='strain',         # NEW: Title mode (even if hidden)
        global_normalization=False,
        num_processes=8
    )
    
    # Parse vertex data
    plotter.parse_time_series_data(f'../{prefix}_{prefix_val}/{en}/data/VertexPositions_N_{N}.dat')

    # Calculate aspect ratios only in the desired strain window
    plotter.calculate_all_aspect_ratios(
        strain_range=(0.09, 0.11),
        # strain_range=(1.01, 1.05),
        # strain_range=(6.0, 7.0),
        # time_range=(2000.0, 4000.0)
    )
    
    # Generate plots with time range (NEW: time_range filtering)
    plotter.generate_plots(
        strain_range=(0.09, 0.11),
        # strain_range=(1.01, 1.05),
        # strain_range=(6.0, 6.1),
        #time_range=(2000.0, 4000.0)  # NEW: Only plot frames in this time window
    )
    
    print(f"\nAll plots saved to {output_dir}/")

