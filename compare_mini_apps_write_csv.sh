#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: mini_apps directory not specified"
    echo "Usage: $0 <mini_apps_dir>"
    exit 1
fi
mini_apps_dir=$1

# Define output file
echo "ng,veclen,rpb,exec_name,cpu_cost,gpu_cost" > performance_table.csv

# Loop through each subfolder in mini_apps
for dir in ${mini_apps_dir}/ng_*; do
    # Extract ng, veclen, and rpb from directory name
    if [[ $dir =~ ng_([0-9]+)_veclen_([0-9]+)_rpb_([0-9]+) ]]; then
        ng=${BASH_REMATCH[1]}
        veclen=${BASH_REMATCH[2]}
        rpb=${BASH_REMATCH[3]}
    else
        continue
    fi

    # Check if "run" directory exists
    run_dir="$dir/run"
    if [ ! -d "$run_dir" ]; then
        continue
    fi

    # Loop through executables inside "run"
    for exec in "$run_dir"/*; do
        if [ -x "$exec" ] && [ ! -d "$exec" ]; then  # Ensure it is an executable file
            exec_name=$(basename "$exec")
            echo $exec 
            # Run the executable and capture the last two lines
            output=$("$exec" 2>&1)

			# Extract CPU and GPU cost from the output using grep and awk
			cpu_cost=$(echo "$output" | grep "CPU Cost (sec):" | awk '{print $4}')
			gpu_cost=$(echo "$output" | grep "GPU Cost (sec):" | awk '{print $4}')

            
            # Append data to the table
            echo "$ng,$veclen,$rpb,$exec_name,$cpu_cost,$gpu_cost" >> performance_table_${mini_apps_dir}.csv
        fi
    done

done

echo "Performance data saved to performance_table.csv"

