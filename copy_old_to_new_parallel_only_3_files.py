import os
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# List of your old roots
#old_roots = ["p0_3.65", "p0_3.7", "p0_3.75", "p0_3.8", "p0_3.82", 
#             "p0_3.83", "p0_3.84", "p0_3.85", "p0_3.86", "p0_3.88", 
#             "p0_3.9", "p0_3.95", "p0_4.0"]

#old_roots = ["p0_3.88", "p0_3.86", "p0_3.85", "p0_3.9", "p0_3.95", "p0_4.0"]
#old_roots = ["p0_3.84",] # "p0_3.83", "p0_3.8"] # 
#old_roots=["p0_3.75", "p0_3.7", "p0_3.65", "p0_3.82"]
old_roots = ["p0_3.79", "p0_3.84", "p0_3.81"]

def copy_files_for_root(old_root):
    new_root = old_root + "_init_T_0.1_ens"
    os.makedirs(new_root, exist_ok=True)
    
    for i in tqdm(range(1, 896+1)):  # 1 to 96 inclusive
        en_name = f"en{i}"
        src_dir = os.path.join(old_root, en_name)
        src_data = os.path.join(src_dir, "data")
        dst_dir = os.path.join(new_root, en_name)

        os.makedirs(dst_dir, exist_ok=True)

        # Copy out*.log files
        for log_file in glob.glob(os.path.join(src_dir, "out*.log")):
            shutil.copy(log_file, dst_dir)

        # Copy Relaxed*, system*, Minimization_energy* from src_data
        #for pattern in ["Relaxed*", "system*", "Minimization_energy*"]:
        #for pattern in ["Relaxed*", "system*"]:
        #    for data_file in glob.glob(os.path.join(src_data, pattern)):
        #        shutil.copy(data_file, dst_dir)

        for filename in [
                "system_A0_P0.dat",
                "Relaxed_Raw_nvv.dat",
                "Relaxed_Raw_positions.dat"
         ]:
            src_file = os.path.join(src_data, filename)
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_dir)


# Use ThreadPoolExecutor to run copying jobs for each old_root in parallel
with ThreadPoolExecutor() as executor:
    list(tqdm(executor.map(copy_files_for_root, old_roots), total=len(old_roots), desc="Copying file sets"))

print("All files copied successfully.")

