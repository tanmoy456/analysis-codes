import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import os
import re
from tqdm import tqdm

# Constants for plot limits (adjust as per your setup)
L = 10
N = L * L
hex_len = np.sqrt(2 / (3 * np.sqrt(3)))
xm = L * np.sqrt(3) * hex_len
ym = L / (np.sqrt(3) * hex_len)

# Function to parse the vertex data with time series from the file
def parse_time_series_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    time_series_data = {}
    current_time = None
    for line in data:
        line = line.strip()
        
        # Detect a new time step
        if line.startswith("Time"):
            current_time = float(line.split(":")[1].strip())
            time_series_data[current_time] = []
        elif line.startswith("Strain"):
            continue  # Ignore strain lines
        elif line:
            row = [float(x) if i > 0 else int(x) for i, x in enumerate(line.split())]
            time_series_data[current_time].append(row)
    
    return time_series_data

# Function to parse area, perimeter, and other cell properties from the file
def parse_area_data(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    area_time_series_data = {}
    current_time = None
    for line in data:
        line = line.strip()
        
        # Detect a new time step
        if line.startswith("Time"):
            current_time = float(line.split(":")[1].strip())
            area_time_series_data[current_time] = []
        elif line.startswith("Strain"):
            continue  # Ignore strain lines
        elif line and not line.startswith("#"):
            row = [float(x) if i > 0 else int(x) for i, x in enumerate(line.split())]
            area_time_series_data[current_time].append(row)
    
    return area_time_series_data

# Function to write a single time step to a VTK file
def write_vtk(filename, time_step_vertex_data, time_step_area_data):
    points = vtk.vtkPoints()
    polygons = vtk.vtkCellArray()
    
    # Arrays to hold data for each cell
    area_data = vtk.vtkFloatArray()
    area_data.SetName("Area")
    
    perimeter_data = vtk.vtkFloatArray()
    perimeter_data.SetName("Perimeter")
    
    vertex_count_data = vtk.vtkIntArray()
    vertex_count_data.SetName("Vertices")
    
    sigma_xx_count_data = vtk.vtkFloatArray()
    sigma_xx_count_data.SetName("Sigma_xx")
    
    sigma_xy_count_data = vtk.vtkFloatArray()
    sigma_xy_count_data.SetName("Sigma_xy")
    
    sigma_yy_count_data = vtk.vtkFloatArray()
    sigma_yy_count_data.SetName("Sigma_yy")

    # Dictionary to store points and cell connectivity
    cells = {}
    
    # Add points and define cells (polygons)
    for row in time_step_vertex_data:
        cell_id = int(row[0])
        x, y = row[1], row[2]
        
        if cell_id not in cells:
            cells[cell_id] = []
        point_id = points.InsertNextPoint([x, y, 0])  # Z is 0 for 2D data
        cells[cell_id].append(point_id)
    
    # Add polygons (cells) to the PolyData structure
    for row in time_step_area_data:
        cell_id = int(row[0])
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(cells[cell_id]))
        for i, point_id in enumerate(cells[cell_id]):
            polygon.GetPointIds().SetId(i, point_id)
        polygons.InsertNextCell(polygon)
        
        # Add area, perimeter, and vertex count for each cell
        area_data.InsertNextValue(row[4])        # Area
        perimeter_data.InsertNextValue(row[5])   # Perimeter
        vertex_count_data.InsertNextValue(int(row[1])) # Number of vertices (convert to integer)
        
        sigma_xx_count_data.InsertNextValue(row[10])   # sigma_xx
        sigma_xy_count_data.InsertNextValue(row[11])   # sigma_xy
        sigma_yy_count_data.InsertNextValue(row[12])   # sigma_yy

    # Create PolyData and add points and polygons
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(polygons)

    # Attach cell data
    poly_data.GetCellData().AddArray(area_data)
    poly_data.GetCellData().AddArray(perimeter_data)
    poly_data.GetCellData().AddArray(vertex_count_data)

    poly_data.GetCellData().AddArray(sigma_xx_count_data)
    poly_data.GetCellData().AddArray(sigma_xy_count_data)
    poly_data.GetCellData().AddArray(sigma_yy_count_data)

    # Write the PolyData to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(poly_data)
    writer.Write()

# Output directory for the VTK files
output_dir = "vtk_time_series"
os.makedirs(output_dir, exist_ok=True)

# Parse the time series data from files
vertex_time_series_data = parse_time_series_data(f'../data/VertexPositions_N_{N}.dat')
area_time_series_data = parse_area_data(f'../data/Cell_propery_N_{N}.dat')

# Loop over each time step and write a VTK file
for time in tqdm(vertex_time_series_data.keys(), desc="Generating VTK files"):
    vertex_data = vertex_time_series_data[time]
    area_data = area_time_series_data[time]
    
    # Create a filename based on the time step
    filename = os.path.join(output_dir, f"cells_frame_{int(time)}.vtk")
    
    # Write the VTK file for the current time step
    write_vtk(filename, vertex_data, area_data)
