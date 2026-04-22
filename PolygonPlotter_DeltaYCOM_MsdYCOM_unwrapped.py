import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Threading control
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


class PolygonPlotterDeltaYCOM:
    """
    Plot polygons colored by |com_yu(t) - com_yu(t=0)| for each cell,
    using UNWRAPPED y-coordinates from Com_All_Cells_Unwrapped_N_*.dat.

    Because unwrapped coordinates are used, boundary-crossing artefacts
    (huge Δy from folded PBC jumps) no longer occur — outlier_threshold
    can be set to None, or kept as a safety net.

    File formats
    ────────────
    Vertex file  : Time:/Strain: header blocks  (unchanged)
    Unwrapped COM: LAMMPS ITEM: blocks
                   columns per data line: id  type  com_xu  com_yu
                   timestep → time via:  time = (timestep - 1) * dt

    The two files must have the same number of frames in the same order.
    """

    def __init__(self, L, dt=0.01,
                 show_blue_box=False, fix_frame=True,
                 global_normalization=True, show_tick_labels=False,
                 show_axis_spines=False, show_title=True,
                 title_mode='strain', save_dpi=100,
                 periodic_tiling=False, crop_to_box=False,
                 cmap_name='viridis', manual_vmax=None,
                 outlier_threshold=None,
                 plot_mode="delta_y"):
        """
        Parameters
        ----------
        dt : float
            Simulation time-step used to convert LAMMPS integer timestep → time.
            time = (timestep - 1) * dt
        outlier_threshold : float or None
            Cells with |Δy_COM| > threshold are skipped entirely (not drawn,
            not used for colorbar). Set None to keep all cells.
        plot_mode : str
            'delta_y' — color by |Δy_COM|          (displacement amplitude)
            'msd_y'   — color by (Δy_COM)²         (local MSD, highlights shear bands)
        """
        if plot_mode not in ('delta_y', 'msd_y'):
            raise ValueError("plot_mode must be 'delta_y' or 'msd_y'.")

        self.L = L
        self.N = L * L
        self.dt = dt
        self.hex_len = np.sqrt(2 / (3 * np.sqrt(3)))
        self.xm = L * np.sqrt(3) * self.hex_len
        self.ym = L / (np.sqrt(3) * self.hex_len)

        self.show_blue_box = show_blue_box
        self.fix_frame = fix_frame
        self.global_normalization = global_normalization
        self.show_tick_labels = show_tick_labels
        self.show_axis_spines = show_axis_spines
        self.show_title = show_title
        self.title_mode = title_mode
        self.save_dpi = save_dpi
        self.periodic_tiling = periodic_tiling
        self.crop_to_box = crop_to_box
        self.cmap = matplotlib.colormaps.get_cmap(cmap_name)
        self.manual_vmax = manual_vmax
        self.outlier_threshold = outlier_threshold
        self.plot_mode = plot_mode   # 'delta_y' or 'msd_y'

        self.vertex_time_series_data = {}   # (time, strain) -> [rows]

        # Unwrapped COM: keyed by (time, strain) matching vertex keys
        # Each entry: dict  cell_id -> com_yu  (float)
        self.unwrapped_com_data = {}

        # Property file kept for strain info only (to build the (time,strain) key)
        self.property_time_series_data = {}

        self.ref_key = None
        self.ref_y_by_cell = {}   # cell_id -> com_yu at reference frame
        self.global_min = 0.0
        self.global_max = 1.0

    # ═══════════════════════════ parsers ════════════════════════════════════

    def _parse_vertex_file(self, filename):
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
                row = [float(x) if i > 0 else int(x)
                       for i, x in enumerate(line.split())]
                data[(current_time, current_strain)].append(row)
        return data

    def _parse_property_file(self, filename):
        """Parse Cell_propery file — used only to obtain (time, strain) keys."""
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
            elif line and not line.startswith("#") and current_time is not None:
                row = [float(x) if i > 0 else int(x)
                       for i, x in enumerate(line.split())]
                data[(current_time, current_strain)].append(row)
        return data

    def _parse_unwrapped_com_file(self, filename, sorted_keys):
        """
        Parse Com_All_Cells_Unwrapped_N_*.dat (LAMMPS ITEM: format).

        Each block:
            ITEM: TIMESTEP
            <int timestep>
            ITEM: NUMBER OF CELLS
            <int N>
            ITEM: BOX BOUNDS pp pp
            <xlo> <xhi>
            <ylo> <yhi>
            ITEM: CELLS id type com_xu com_yu
            <id> <type> <com_xu> <com_yu>
            ...

        Returns dict: (time, strain) -> {cell_id: com_yu}
        The (time, strain) keys are matched to sorted_keys by frame order.
        time is recovered as (timestep - 1) * dt.
        """
        blocks = []          # list of {timestep, cells: {id: com_yu}}
        current_block = None
        reading_cells = False

        with open(filename, "r") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "ITEM: TIMESTEP":
                if current_block is not None:
                    blocks.append(current_block)
                current_block = {'timestep': None, 'cells': {}}
                reading_cells = False
                i += 1
                current_block['timestep'] = int(lines[i].strip())

            elif line == "ITEM: NUMBER OF CELLS":
                reading_cells = False  # skip next line

            elif line.startswith("ITEM: BOX BOUNDS"):
                reading_cells = False

            elif line.startswith("ITEM: CELLS"):
                # "ITEM: CELLS id type com_xu com_yu"
                reading_cells = True

            elif reading_cells and line and not line.startswith("ITEM:"):
                parts = line.split()
                if len(parts) >= 4:
                    cell_id = int(parts[0])
                    com_yu  = float(parts[3])
                    current_block['cells'][cell_id] = com_yu

            i += 1

        if current_block is not None:
            blocks.append(current_block)

        print(f"  Parsed {len(blocks)} blocks from unwrapped COM file.")

        if len(blocks) != len(sorted_keys):
            print(f"  WARNING: {len(blocks)} unwrapped blocks vs "
                  f"{len(sorted_keys)} vertex/property keys — matching by order.")

        # Match blocks → (time, strain) keys by frame order
        data = {}
        for idx, block in enumerate(blocks):
            if idx >= len(sorted_keys):
                break
            key = sorted_keys[idx]
            data[key] = block['cells']   # dict: cell_id -> com_yu

        return data

    # ═══════════════════════════ entry point ════════════════════════════════

    def parse_time_series_data(self, vertex_filename, property_filename,
                               unwrapped_com_filename):
        """
        Load all three files and build reference frame.

        Parameters
        ----------
        vertex_filename        : path to VertexPositions_N_*.dat
        property_filename      : path to Cell_propery_N_*.dat  (for strain keys)
        unwrapped_com_filename : path to Com_All_Cells_Unwrapped_N_*.dat
        """
        self.vertex_time_series_data = self._parse_vertex_file(vertex_filename)
        self.property_time_series_data = self._parse_property_file(property_filename)

        if not self.vertex_time_series_data:
            raise ValueError("No vertex data found.")
        if not self.property_time_series_data:
            raise ValueError("No property data found.")

        sorted_keys = sorted(self.vertex_time_series_data.keys())
        self.unwrapped_com_data = self._parse_unwrapped_com_file(
            unwrapped_com_filename, sorted_keys)

        if not self.unwrapped_com_data:
            raise ValueError("No unwrapped COM data parsed.")

        self.ref_key = self._find_reference_key()
        self.ref_y_by_cell = self._build_ref_y_map(self.ref_key)
        # NOTE: vmin/vmax are computed in generate_plots after the range is known

    # ═══════════════════════════ reference ══════════════════════════════════

    def _find_reference_key(self):
        keys = sorted(self.unwrapped_com_data.keys())
        tol = 1e-10
        for k in keys:
            if abs(k[0]) < tol and abs(k[1]) < tol:
                return k
        earliest = min(keys, key=lambda k: (k[0], k[1]))
        print(f"  Reference: exact (0,0) not found; using earliest key {earliest}")
        return earliest

    def _build_ref_y_map(self, ref_key):
        ref_cells = self.unwrapped_com_data.get(ref_key, {})
        if not ref_cells:
            raise ValueError(f"Reference frame {ref_key} has no unwrapped COM-y data.")
        print(f"  Reference frame: time={ref_key[0]}, strain={ref_key[1]}  "
              f"({len(ref_cells)} cells)")
        return dict(ref_cells)   # cell_id -> com_yu at t=0

    # ═══════════════════════════ global bounds ═══════════════════════════════

    def _compute_global_bounds(self, selected_keys):
        """Compute vmin/vmax only over the selected frames (not all frames)."""
        all_vals = []
        for key in selected_keys:
            cell_yu_map = self.unwrapped_com_data.get(key, {})
            for cell_id, yu in cell_yu_map.items():
                if cell_id in self.ref_y_by_cell:
                    delta = abs(yu - self.ref_y_by_cell[cell_id])
                    val = delta ** 2 if self.plot_mode == 'msd_y' else delta
                    all_vals.append(val)

        if not all_vals:
            return 0.0, 1.0

        if self.outlier_threshold is not None:
            outliers = sorted([v for v in all_vals if v > self.outlier_threshold], reverse=True)
            bulk     = [v for v in all_vals if v <= self.outlier_threshold]

            print(f"\n{'='*58}")
            print(f"  outlier_threshold = {self.outlier_threshold}")
            print(f"  Cells ABOVE threshold (skipped): {len(outliers)}")
            print(f"{'='*58}")
            for rank, v in enumerate(outliers, start=1):
                print(f"  Rank {rank:4d}:  |Δy_COM| = {v:.6f}")
            print(f"{'='*58}")
            print(f"  Cells kept (bulk): {len(bulk)}")
            print(f"{'='*58}\n")
            bulk_vals = bulk if bulk else all_vals
        else:
            bulk_vals = all_vals

        vmin = float(min(bulk_vals))
        vmax = float(max(bulk_vals)) if self.manual_vmax is None else float(self.manual_vmax)

        if vmax <= vmin:
            vmax = vmin + 1.0
        if abs(vmax - vmin) < 1e-14:
            vmax = vmin + 1.0

        print(f"{'='*58}")
        print(f"  plot_mode = '{self.plot_mode}'")
        print(f"  Colorbar range (unwrapped, bulk cells only):")
        print(f"    vmin = {vmin:.6f}  →  darkest  color ({self.cmap.name})")
        print(f"    vmax = {vmax:.6f}  →  brightest color ({self.cmap.name})")
        if self.outlier_threshold is not None:
            print(f"  Outlier cells (|Δy| > {self.outlier_threshold}) not drawn.")
        print(f"{'='*58}\n")

        return vmin, vmax

    # ═══════════════════════════ single frame ════════════════════════════════

    def plot_frame(self, args):
        (time, strain, vtx_data, cell_yu_map, frame_number, bounds,
         show_blue_box, fix_frame, output_dir, global_norm,
         show_tick_labels, show_axis_spines, show_title, title_mode,
         save_dpi, periodic_tiling, crop_to_box, ref_y_by_cell,
         cmap_name, manual_vmax, outlier_threshold, plot_mode) = args

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = matplotlib.colormaps.get_cmap(cmap_name)

        # Build polygon vertex dict
        cells = {}
        xlo, xhi = 0.0, bounds[2]
        ylo, yhi = 0.0, bounds[3]
        for row in vtx_data:
            cell, x, y = int(row[0]), row[1], row[2]
            if cell not in cells:
                cells[cell] = []
            cells[cell].append((x, y))

        if periodic_tiling:
            shifts = [
                (-bounds[2], -bounds[3]), (-bounds[2], 0.0), (-bounds[2], bounds[3]),
                (0.0, -bounds[3]), (0.0, 0.0), (0.0, bounds[3]),
                (bounds[2], -bounds[3]), (bounds[2], 0.0), (bounds[2], bounds[3]),
            ]
        else:
            shifts = [(0.0, 0.0)]

        def intersects_box(coords):
            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]
            return not (max(xs) < xlo or min(xs) > xhi or max(ys) < ylo or min(ys) > yhi)

        # Build delta_by_cell from UNWRAPPED y-positions
        # outlier check always on |Δy| (before squaring)
        delta_by_cell = {}
        for cell_id, yu_now in cell_yu_map.items():
            ref_y = ref_y_by_cell.get(cell_id)
            if ref_y is None:
                continue
            delta = abs(yu_now - ref_y)
            if outlier_threshold is not None and delta > outlier_threshold:
                continue   # skip outlier entirely
            val = delta ** 2 if plot_mode == 'msd_y' else delta
            delta_by_cell[cell_id] = val

        # Normalization
        if global_norm:
            norm = mcolors.Normalize(vmin=bounds[0], vmax=bounds[1])
        else:
            frame_vals = list(delta_by_cell.values())
            if frame_vals:
                vmin_f = min(frame_vals)
                vmax_f = max(frame_vals) if manual_vmax is None else float(manual_vmax)
                if vmax_f <= vmin_f:
                    vmax_f = vmin_f + 1.0
                if abs(vmax_f - vmin_f) < 1e-14:
                    vmax_f = vmin_f + 1.0
                norm = mcolors.Normalize(vmin=vmin_f, vmax=vmax_f)
            else:
                norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        # Draw polygons
        for cell, base_coords in cells.items():
            if cell not in delta_by_cell:
                continue
            color_value = delta_by_cell[cell]
            for dx, dy in shifts:
                shifted_coords = [(x + dx, y + dy) for x, y in base_coords]
                if periodic_tiling and not intersects_box(shifted_coords):
                    continue
                poly = Polygon(shifted_coords, closed=True,
                               edgecolor='black',
                               facecolor=cmap(norm(color_value)),
                               linewidth=0.3)
                ax.add_patch(poly)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_label = (r'$(\Delta y_{\mathrm{COM}})^2$' if plot_mode == 'msd_y'
                      else r'$|\Delta y_{\mathrm{COM}}|$')
        plt.colorbar(sm, label=cbar_label, ax=ax, shrink=0.8)

        if show_blue_box:
            ax.plot([0, bounds[2], bounds[2], 0, 0],
                    [0, 0, bounds[3], bounds[3], 0],
                    'b-', linewidth=1)

        if crop_to_box:
            ax.set_xlim(0, bounds[2])
            ax.set_ylim(0, bounds[3])
        elif fix_frame:
            ax.set_xlim(-2, bounds[2] + 2)
            ax.set_ylim(-2, bounds[3] + 2)

        if show_title:
            if title_mode == 'strain':
                ax.set_title(rf'$\gamma = {strain:.6f}$', fontsize=10)
            elif title_mode == 'time':
                ax.set_title(rf'$t = {time:.4f}$', fontsize=10)

        if not show_tick_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', which='both', length=0)
        if not show_axis_spines:
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.savefig(f"{output_dir}/snapshot_{plot_mode}_frame_{frame_number:05d}.png",
                    bbox_inches="tight", pad_inches=0.01,
                    dpi=save_dpi, facecolor='white')
        plt.close(fig)

    # ═══════════════════════════ driver ══════════════════════════════════════

    def generate_plots(self, output_dir, num_processes=8,
                       strain_range=None, time_range=None):
        os.makedirs(output_dir, exist_ok=True)

        all_keys = sorted(self.vertex_time_series_data.keys())

        if strain_range is not None and time_range is not None:
            raise ValueError("Use either strain_range or time_range, not both.")

        if strain_range is not None:
            selected_keys = [k for k in all_keys
                             if strain_range[0] <= k[1] <= strain_range[1]]
        elif time_range is not None:
            selected_keys = [k for k in all_keys
                             if time_range[0] <= k[0] <= time_range[1]]
        else:
            selected_keys = all_keys

        if not selected_keys:
            raise ValueError(f"No frames found for given range: "
                             f"strain_range={strain_range}, time_range={time_range}")

        # Compute vmin/vmax only from the selected range of frames
        self.global_min, self.global_max = self._compute_global_bounds(selected_keys)
        bounds = (self.global_min, self.global_max, self.xm, self.ym)

        args_list = []
        for i, key in enumerate(selected_keys):
            args_list.append((
                key[0], key[1],
                self.vertex_time_series_data[key],
                self.unwrapped_com_data.get(key, {}),   # cell_id -> com_yu
                i + 1, bounds,
                self.show_blue_box, self.fix_frame,
                output_dir, self.global_normalization,
                self.show_tick_labels, self.show_axis_spines,
                self.show_title, self.title_mode,
                self.save_dpi, self.periodic_tiling,
                self.crop_to_box, self.ref_y_by_cell,
                self.cmap.name, self.manual_vmax,
                self.outlier_threshold,
                self.plot_mode,
            ))

        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            list(tqdm(pool.map(self.plot_frame, args_list),
                      total=len(args_list),
                      desc=f"Generating {self.plot_mode} frames"))


# ═══════════════════════════════ main ════════════════════════════════════════

if __name__ == "__main__":
    prefix       = 'gd'
    prefix_value = '0.0001'
    en           = 'en4'
    path         = f'../{prefix}_{prefix_value}/{en}/data/'

    L = 20
    N = L * L

    output_dir = f"config_figures_{prefix}_{prefix_value}_{en}_delta_ycom"

    plotter = PolygonPlotterDeltaYCOM(
        L,
        dt=0.01,                     # time = (timestep - 1) * dt
        global_normalization=True,
        show_blue_box=False,
        show_tick_labels=False,
        show_axis_spines=False,
        show_title=True,
        title_mode='strain',
        save_dpi=100,
        periodic_tiling=False,
        crop_to_box=False,
        cmap_name='viridis',
        # manual_vmax=5.0,           # fix colorbar max if needed
        outlier_threshold=None,      # unwrapped coords: no boundary jumps expected
        plot_mode='msd_y',           # 'delta_y' → |Δy|  |  'msd_y' → (Δy)²
    )

    plotter.parse_time_series_data(
        vertex_filename          = f'{path}VertexPositions_N_{N}.dat',
        property_filename        = f'{path}Cell_propery_N_{N}.dat',
        unwrapped_com_filename   = f'{path}Com_All_Cells_Unwrapped_N_{N}.dat',
    )

    plotter.generate_plots(
        output_dir,
        num_processes=8,
        strain_range=(4.0, 8.0),
        # time_range=(40000.0, 48000.0),
    )
