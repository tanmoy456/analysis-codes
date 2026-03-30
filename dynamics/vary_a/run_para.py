import subprocess
import multiprocessing

def run_script(arg):
    #subprocess.run(["/home/gtanmoy/miniconda3/bin/python3", "msd_qt_chi4_y_data.py", str(arg)])
    subprocess.run(["/home/gtanmoy/miniconda3/bin/python3", "msd_corr_chi4_y_data.py", str(arg)])
    #subprocess.run(["/home/gtanmoy/miniconda3/bin/python3", "msd_fs_chi4_y_data.py", str(arg)])

if __name__ == "__main__":
    arguments = ["0.001", "0.0001"]
    with multiprocessing.Pool(processes=2) as pool:
        pool.map(run_script, arguments)

