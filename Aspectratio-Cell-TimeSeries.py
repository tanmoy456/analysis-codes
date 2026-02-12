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
        """Calculate the centroid of a polygon using the shoelace formula."""
        x = vertices[:, 0]
        y = vertices[:, 1]
        
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        
        cross = x * y_next - x_next * y
        A = 0.5 * np.sum(cross)
        
        if abs(A) < 1e-10:
            # Degenerate polygon, return geometric mean
            return np.mean(x), np.mean(y), 0.0
        
        cx = (1/(6*A)) * np.sum((x + x_next) * cross)
        cy = (1/(6*A)) * np.sum((y + y_next) * cross)
        
        return cx, cy, abs(A)
    
    def polygon_inertia_centroidal(self, vertices):
        """
        Calculate centroidal moment of inertia using the polygon formula.
        
        Steps:
        1. Calculate centroid
        2. Shift vertices to centroid frame
        3. Calculate inertia tensor
        """
        vertices = np.array(vertices)
        
        # Calculate centroid and area
        cx, cy, area = self.polygon_centroid(vertices)
        
        if area < 1e-10:
            # Degenerate polygon
            return 0.0, 0.0, 0.0, area
        
        # Shift to centroid frame
        x_cent = vertices[:, 0] - cx
        y_cent = vertices[:, 1] - cy
        
        # Cyclic indexing for closed polygon
        x_cent_next = np.roll(x_cent, -1)
        y_cent_next = np.roll(y_cent, -1)
        
        # Cross product term: (xi*yi+1 - xi+1*yi)
        cross_cent = x_cent * y_cent_next - x_cent_next * y_cent
        
        # Calculate Iy (second moment about y-axis)
        Iy = (1/12) * np.sum(cross_cent * (x_cent**2 + x_cent*x_cent_next + x_cent_next**2))
        
        # Calculate Ix (second moment about x-axis)
        Ix = (1/12) * np.sum(cross_cent * (y_cent**2 + y_cent*y_cent_next + y_cent_next**2))
        
        # Calculate Ixy (product of inertia)
        Ixy = (1/24) * np.sum(cross_cent * (x_cent*y_cent_next + 2*x_cent*y_cent + 
                                             2*x_cent_next*y_cent_next + x_cent_next*y_cent))
        
        return Ix, Iy, Ixy, area
    
    def compute_aspect_ratio(self, Ix, Iy, Ixy):
        """Compute aspect ratio from inertia tensor eigenvalues."""
        inertia_tensor = np.array([[Ix, Ixy], 
                                   [Ixy, Iy]])
        
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        eigenvalues = np.abs(eigenvalues)
        
        # Avoid division by zero
        if eigenvalues[0] < 1e-10:
            return 1.0
        
        lambda_min, lambda_max = np.sort(eigenvalues)
        aspect_ratio = math.sqrt(lambda_max / lambda_min)
        
        return aspect_ratio
    
    def calculate_all_aspect_ratios(self):
        """Calculate aspect ratios for all cells at all time steps."""
        print("\nCalculating aspect ratios for all time steps...")
        
        for (time, strain), vertex_data in tqdm(self.vertex_time_series_data.items(), 
                                                 desc="Processing time steps"):
            # Organize vertices by cell
            cells = {}
            for row in vertex_data:
                cell = int(row[0])
                x, y = row[1], row[2]
                if cell not in cells:
                    cells[cell] = []
                cells[cell].append([x, y])
            
            # Calculate aspect ratio for each cell
            time_step_aspect_ratios = {}
            for cell, vertices in cells.items():
                vertices = np.array(vertices)
                
                # Remove duplicate last vertex if it equals the first
                if len(vertices) > 1 and np.allclose(vertices[0], vertices[-1]):
                    vertices = vertices[:-1]
                
                # Skip if too few vertices
                if len(vertices) < 3:
                    continue
                
                # Compute centroidal inertia tensor using polygon formula
                Ix, Iy, Ixy, area = self.polygon_inertia_centroidal(vertices)
                
                # Skip degenerate polygons
                if area < 1e-10:
                    continue
                
                # Compute aspect ratio
                aspect_ratio = self.compute_aspect_ratio(Ix, Iy, Ixy)
                
                time_step_aspect_ratios[cell] = aspect_ratio
            
            self.aspect_ratios[(time, strain)] = time_step_aspect_ratios
    
    def save_aspect_ratios(self, output_filename):
        """Save aspect ratios to a file."""
        with open(output_filename, 'w') as f:
            f.write("# Time Strain Cell AspectRatio\n")
            
            for (time, strain), cell_aspect_ratios in sorted(self.aspect_ratios.items()):
                for cell, aspect_ratio in sorted(cell_aspect_ratios.items()):
                    f.write(f"{time:.6f} {strain:.8f} {cell} {aspect_ratio:.6f}\n")
        
        print(f"\nAspect ratios saved to {output_filename}")
    
    def get_aspect_ratio_statistics(self):
        """Print statistics of aspect ratios."""
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


# Usage
if __name__ == '__main__':

    # Parent directory for all outputs
    parent_dir = "AspectRatioResults"
    os.makedirs(parent_dir, exist_ok=True)

    prefix = 'gd'
    prefix_val = '0.00001'

    # Create subfolder for this prefix_val
    output_dir = os.path.join(parent_dir, f'AspectRatio_{prefix}_{prefix_val}')
    os.makedirs(output_dir, exist_ok=True)

    L = 10
    N = L * L
    
    # Loop over en1 to en8
    for en_num in range(1, 9):
        en = f'en{en_num}'
        
        print(f"\n{'='*60}")
        print(f"Processing {en} for {prefix}_{prefix_val}")
        print(f"{'='*60}")
        
        calculator = PolygonInertiaCalculator(L=L)
        
        # Parse vertex data
        input_file = f'../{prefix}_{prefix_val}/{en}/data/VertexPositions_N_{N}.dat'
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} does not exist. Skipping {en}.")
            continue
        
        calculator.parse_time_series_data(input_file)
        
        # Calculate aspect ratios
        calculator.calculate_all_aspect_ratios()
        
        # Save to file in the output subdirectory
        output_file = os.path.join(output_dir, f'aspect_ratios_{prefix}_{prefix_val}_{en}_N_{N}.dat')
        calculator.save_aspect_ratios(output_file)
        
        # Print statistics
        calculator.get_aspect_ratio_statistics()
    
    print(f"\n{'='*60}")
    print(f"All 8 ensemble files saved in: {output_dir}")
    print(f"{'='*60}")
