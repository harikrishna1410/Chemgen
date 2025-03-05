#!/bin/bash
veclen_arr=(16 32 64 128)
rpb_arr=(16 8 4 2)

# Check if CSV file argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a CSV file as argument"
    exit 1
fi

# Loop from 0 to 3 (inclusive) to iterate through array indices
for i in $(seq 0 3); do
    python plot_omp_speedup.py --veclen ${veclen_arr[$i]} --rpb ${rpb_arr[$i]} --csv_name $1
done
