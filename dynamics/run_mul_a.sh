#!/bin/bash

# List of 'a' values
a_values=("0.001" "0.002" "0.003" "0.004" "0.005")

# Loop over each 'a' and run the script
for a in "${a_values[@]}"
do
    echo "Running script for a = $a"
    ~/miniconda3/bin/python3 tau_alpha.py "$a"
    ~/miniconda3/bin/python3 chi_peak.py "$a"
done
