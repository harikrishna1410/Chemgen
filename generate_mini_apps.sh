#!/bin/bash
ng=(64 256 1024 4096 16384 65536 262144)
veclen=(16 32 64 128)
rpb=(16 8 4 2)
if [ -z "$1" ]; then
    echo "Error: Chemistry file argument is required"
    echo "Usage: $0 <chemistry_directory>"
    exit 1
fi
chem=$1

mkdir -p mini_apps
for ng in ${ng[@]};do
    for i in $(seq 0 3);do
        echo "veclen=${veclen[$i]} rpb=${rpb[$i]} ng=$ng"
        dir=mini_apps/ng_${ng}_veclen_${veclen[$i]}_rpb_${rpb[$i]}
        mkdir -p $dir
        cp mini_app/CMakeLists.txt $dir/
        python3 generate_chemistry.py --mode mini_app --mech ${chem}/chem.inp --therm ${chem}/therm.dat --output $dir --time-cpu --ng 64 --nreact-per-block ${rpb[$i]} --veclen ${veclen[$i]} --ng $ng
    done
done