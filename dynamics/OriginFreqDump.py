import re
import os
import sys
import numpy as np

tf=1e6
#file = "DumpFreq_tw_10000000_pts_80.txt" # 1e7 & 80 frames
file = f"DumpFreq_tw_{int(tf)}_pts_80.txt"  # 1e6 & 80 frames
# Use regular expression to extract the numbers
result = re.findall(r'\d+', file)
# Convert the extracted numbers to integers
values = [int(num) for num in result]
tw = values[0]
print("time window:", tw , "and number of frames in one time window: ", values[1])

# read the data file
data = np.loadtxt(file)

origins = 10
origin_diff = 1e6
# make folder if not exists
folder_name = f"origin_files_{origins}_od_{int(origin_diff)}_tf_{int(tf)}"
# Check if the folder already exists
if not os.path.exists(folder_name):
    # Create the folder
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully!")
else:
    print(f"Folder '{folder_name}' already exists.")

# define origins number and difference b/w origins
#origins = 100
frames = values[1] * origins
#origin_diff = 1e6
allframes = []
for time_ave in range(origins):
    temp = data + (time_ave) *  origin_diff
    allframes.append(temp)
    np.savetxt(f"{folder_name}/origin_{(time_ave + 1):04d}.txt", temp, fmt="%d")
     
allframes = np.array(allframes)
allframes = allframes.reshape(-1)
sorted_result = np.sort(allframes)
unique_result = np.unique(sorted_result)

print(f"total number frames in {origins} origins :" ,unique_result.shape[0])
print("last frame : ", unique_result[-1])

unique_values, counts = np.unique(sorted_result, return_counts=True)
repeated_values = unique_values[counts > 1]
print("number of repeated frames:",repeated_values.shape[0])

expected_frames = values[1] * origins
missing_frames = expected_frames - unique_result.shape[0]
if missing_frames > 0:
    print(f"Adding {missing_frames} missing frames.")
    zeros_to_add = np.zeros(missing_frames, dtype=int)
    unique_result = np.concatenate([unique_result, zeros_to_add])

#np.savetxt(f"{folder_name}/Dump_Nstep_{tw}_origins_{origins}_pts_{expected_frames}.txt", unique_result, fmt='%d')

np.savetxt(f"{folder_name}/Dump_Nstep_{tw}_od_{int(origin_diff)}_origins_{origins}_pts_{unique_result.shape[0]}.txt",unique_result,fmt='%d')

print("Files saved!")

