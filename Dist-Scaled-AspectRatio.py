import numpy as np
import matplotlib.pyplot as plt
import os

# Parent directory where aspect ratio results are stored
parent_dir = "AspectRatioResults"

prefix = 'gd'
prefix_val = '0.1'
ss_cutoff = 8

# Subfolder containing all ensemble files
input_dir = os.path.join(parent_dir, f'AspectRatio_{prefix}_{prefix_val}')

# Output directory for histograms
output_dir = os.path.join(parent_dir, f'Histograms_Scaled_aspectRatio_{prefix}')
os.makedirs(output_dir, exist_ok=True)

# Collect all aspect ratios from all ensembles
all_aspect_ratios = []

print(f"Reading ensemble files from: {input_dir}")

# Loop through all ensemble files (en1 to en32)
for en_num in range(1, 8):
    en = f'en{en_num}'
    filename = os.path.join(input_dir, f'aspect_ratios_{prefix}_{prefix_val}_{en}_N_100.dat')

    # Check if file exists
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Skipping {en}.")
        continue

    try:
        # Load the data (columns: Time, Strain, Cell, AspectRatio)
        data = np.loadtxt(filename, comments="#")

        # Filter by strain > 6 (steady state)
        filtered_data = data[data[:, 1] > ss_cutoff]

        # Extract aspect ratios (last column)
        aspect_ratios = filtered_data[:, -1]
        # scaled aspect ratio
        #scaled_aspect_ratios = (filtered_data[:, -1] - 1)/(np.mean(filtered_data[:, -1]) - 1)
        # Append to combined list
        all_aspect_ratios.extend(aspect_ratios)
        #all_aspect_ratios.extend(scaled_aspect_ratios)

        print(f"Loaded {len(aspect_ratios)} aspect ratios from {en}")

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

# Convert to numpy array
all_aspect_ratios = np.array(all_aspect_ratios)
all_aspect_ratios = (all_aspect_ratios - 1) / (np.mean(all_aspect_ratios) - 1)

print(f"\nTotal aspect ratios collected: {len(all_aspect_ratios)}")
print(f"Mean: {np.mean(all_aspect_ratios):.4f}")
print(f"Std: {np.std(all_aspect_ratios):.4f}")
print(f"Min: {np.min(all_aspect_ratios):.4f}")
print(f"Max: {np.max(all_aspect_ratios):.4f}")

# Compute histogram
bins = 40
counts, bin_edges = np.histogram(all_aspect_ratios, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Save histogram data
histogram_file = os.path.join(output_dir, f'histogram_scaled_aspect_ratio_{prefix}_{prefix_val}_all_ensembles_ss.dat')
np.savetxt(
    histogram_file,
    np.column_stack([bin_centers, counts]),
    header="BinCenter Count",
    fmt='%.6f'
)

print(f"\nHistogram data saved to: {histogram_file}")

# Plot histogram
# plt.figure(figsize=(10, 6))
# plt.hist(all_aspect_ratios, bins=bins, density=True, edgecolor='black', alpha=0.7, color='steelblue')
# plt.axvline(np.mean(all_aspect_ratios), color='red', linestyle='--', linewidth=2,
#             label=f'Mean: {np.mean(all_aspect_ratios):.3f}')
# plt.axvline(np.median(all_aspect_ratios), color='green', linestyle='--', linewidth=2,
#             label=f'Median: {np.median(all_aspect_ratios):.3f}')
# plt.xlabel('Aspect Ratio', fontsize=12)
# plt.ylabel('Probability Density', fontsize=12)
# plt.title(f'Aspect Ratio Distribution - {prefix}={prefix_val} (All Ensembles, Strain > 6)', fontsize=14)
# plt.legend(fontsize=10)
# plt.grid(True, alpha=0.3)

# # Add statistics text box
# textstr = f'N samples: {len(all_aspect_ratios)}\nMean: {np.mean(all_aspect_ratios):.3f}\nStd: {np.std(all_aspect_ratios):.3f}'
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# plt.text(0.75, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
#          verticalalignment='top', bbox=props)

# plt.tight_layout()

# # Save plot
# plot_file = os.path.join(output_dir, f'histogram_{prefix}_{prefix_val}_all_ensembles_ss.png')
# plt.savefig(plot_file, dpi=150, bbox_inches='tight')
# print(f"Histogram plot saved to: {plot_file}")

# plt.show()

