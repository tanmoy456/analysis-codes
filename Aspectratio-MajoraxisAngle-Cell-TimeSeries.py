import os
import numpy as np
import math
from tqdm import tqdm

class PolygonInertiaCalculator:
    def __init__(self, L):
        self.L = L
        self.N = L * L
        self.vertex_time_series_data = {}
        self.aspect_ratios = {}
        self.major_axis_angles = {}
        
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
    
    def polygon_centroid(self, vertices):
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
        vertices = np.array(vertices)
        cx, cy, area = self.polygon_centroid(vertices)
        if area < 1e-10:
            return 0.0, 0.0, 0.0, area
        x_cent = vertices[:, 0] - cx
        y_cent = vertices[:, 1] - cy
        x_cent_next = np.roll(x_cent, -1)
        y_cent_next = np.roll(y_cent, -1)
        cross_cent = x_cent * y_cent_next - x_cent_next * y_cent
        Iy = (1/12) * np.sum(cross_cent * (x_cent**2 + x_cent*x_cent_next + x_cent_next**2))
        Ix = (1/12) * np.sum(cross_cent * (y_cent**2 + y_cent*y_cent_next + y_cent_next**2))
        Ixy = (1/24) * np.sum(cross_cent * (x_cent*y_cent_next + 2*x_cent*y_cent +
                                             2*x_cent_next*y_cent_next + x_cent_next*y_cent))
        return Ix, Iy, Ixy, area
    
    def compute_aspect_ratio_and_angle(self, Ix, Iy, Ixy):
        # correct assignment for the inertia tensor
        inertia_tensor = np.array([[Iy, Ixy], [Ixy, Ix]])
        eigvals, eigvecs = np.linalg.eigh(inertia_tensor)
        eigvals = np.abs(eigvals)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvals[1] < 1e-10:
            aspect_ratio = 1.0
        else:
            aspect_ratio = np.sqrt(eigvals[0]/eigvals[1])
        major_axis_vec = eigvecs[:, 0]
        angle = np.arctan2(major_axis_vec[1], major_axis_vec[0])
        return aspect_ratio, angle

    def calculate_all_aspect_ratios_and_angles(self):
        print("\nCalculating aspect ratios and angles for all time steps...")
        for (time, strain), vertex_data in tqdm(self.vertex_time_series_data.items(),
                                                desc="Processing time steps"):
            cells = {}
            for row in vertex_data:
                cell = int(row[0])
                x, y = row[1], row[2]
                if cell not in cells:
                    cells[cell] = []
                cells[cell].append([x, y])
            aspect_ratios = {}
            angles = {}
            for cell, vertices in cells.items():
                vertices = np.array(vertices)
                if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
                    vertices = vertices[:-1]
                if len(vertices) < 3:
                    continue
                Ix, Iy, Ixy, area = self.polygon_inertia_centroidal(vertices)
                if area < 1e-10:
                    continue
                aspect_ratio, angle = self.compute_aspect_ratio_and_angle(Ix, Iy, Ixy)
                aspect_ratios[cell] = aspect_ratio
                angles[cell] = angle
            self.aspect_ratios[(time, strain)] = aspect_ratios
            self.major_axis_angles[(time, strain)] = angles

    def save_aspect_ratios_and_angles(self, output_filename):
        with open(output_filename, 'w') as f:
            f.write("# Time Strain Cell AspectRatio MajorAxisAngleDeg\n")
            for (time, strain), cell_aspect_ratios in sorted(self.aspect_ratios.items()):
                angles = self.major_axis_angles[(time, strain)]
                for cell in sorted(cell_aspect_ratios.keys()):
                    # Save the angle in degrees (range [0, 180))
                    angle_deg = np.degrees(angles[cell]) % 180
                    f.write(f"{time:.6f} {strain:.8f} {cell} {cell_aspect_ratios[cell]:.6f} {angle_deg:.6f}\n")
        print(f"Aspect ratios and angles (degrees) saved to {output_filename}")

    def get_aspect_ratio_statistics(self):
        all_aspect_ratios = []
        for cell_aspect_ratios in self.aspect_ratios.values():
            all_aspect_ratios.extend(cell_aspect_ratios.values())
        if len(all_aspect_ratios) == 0:
            print("No aspect ratios calculated!")
            return
        all_aspect_ratios = np.array(all_aspect_ratios)
        print("\n=== Aspect Ratio Statistics ===")
        print(f"Mean: {np.mean(all_aspect_ratios):.4f}")
        print(f"Std: {np.std(all_aspect_ratios):.4f}")
        print(f"Min: {np.min(all_aspect_ratios):.4f}")
        print(f"Max: {np.max(all_aspect_ratios):.4f}")
        print(f"Median: {np.median(all_aspect_ratios):.4f}")

# ***8 Usage ****
if __name__ == '__main__':
    parent_dir = "AspectRatioResults"
    os.makedirs(parent_dir, exist_ok=True)
    prefix = 'gd'
    prefix_val = '0.00001'
    output_dir = os.path.join(parent_dir, f'AspectRatio_Major_axis_Angle_{prefix}_{prefix_val}')
    os.makedirs(output_dir, exist_ok=True)
    L = 10
    N = L * L
    #for en_num in range(1, 8):  # use range(1, 33) for all ensembles
    for en_num in range(9, 16):  # use range(1, 33) for all ensembles
        en = f'en{en_num}'
        print(f"\n{'='*60}")
        print(f"Processing {en} for {prefix}_{prefix_val}")
        print(f"{'='*60}")
        calculator = PolygonInertiaCalculator(L=L)
        input_file = f'../{prefix}_{prefix_val}/{en}/data/VertexPositions_N_{N}.dat'
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} does not exist. Skipping {en}.")
            continue
        calculator.parse_time_series_data(input_file)
        calculator.calculate_all_aspect_ratios_and_angles()
        output_file = os.path.join(output_dir, f'aspect_ratios_{prefix}_{prefix_val}_{en}_N_{N}.dat')
        calculator.save_aspect_ratios_and_angles(output_file)
        calculator.get_aspect_ratio_statistics()
    print(f"\n{'='*60}")
    print(f"Ensemble files with angles saved in: {output_dir}")
    print(f"{'='*60}")
