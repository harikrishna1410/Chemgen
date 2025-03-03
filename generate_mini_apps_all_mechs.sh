#!/bin/bash
ng=(64 256 1024 4096 16384 65536 262144)
ng=(64)
veclen=(16 32 64 128)
rpb=(16 8 4 2)
# if [ -z "$1" ]; then
#     echo "Error: Chemistry file argument is required"
#     echo "Usage: $0 <chemistry_directory>"
#     exit 1
# fi

chem_files=$(find ./ck_files -name "*.yaml")
chem_name=$(basename "$chem")

for chem in ${chem_files[@]};do
    chem_name=$(basename $(dirname "$chem"))
    echo "Generating mini apps for $chem_name"
    mkdir -p "mini_apps_${chem_name}"
    for ng in ${ng[@]};do
        for i in $(seq 0 0);do
            echo "veclen=${veclen[$i]} rpb=${rpb[$i]} ng=$ng"
            dir="mini_apps_${chem_name}/ng_${ng}_veclen_${veclen[$i]}_rpb_${rpb[$i]}"
            mkdir -p $dir
            cp mini_app/CMakeLists.txt $dir/
            python3 generate_chemistry.py --mode mini_app --yaml_file "${chem}" --rtypes-together --output $dir --time-cpu --ng 64 --nreact-per-block ${rpb[$i]} --veclen ${veclen[$i]} --ng $ng
        done
    done
done

