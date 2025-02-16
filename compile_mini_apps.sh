#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: mini_apps directory not specified"
    echo "Usage: $0 <mini_apps_dir>"
    exit 1
fi
mini_apps_dir=$1
chem=$1

cd $mini_apps_dir

dirs=(*)
for dir in ${dirs[@]};do
    if [ -d $dir ];then
        if [ -f $dir/CMakeLists.txt ];then
            echo "Building $dir"
            cd $dir
            mkdir -p build
            mkdir -p run
            cd build
            ##make omp
            cmake ..
            make -j
            mv ../run/mini_app ../run/mini_app_omp
            ##make HIP
            for PARELLEL_LEVEL in 1 2 3 4;do
                make clean
                cmake -DPARALLEL_LEVEL=$PARELLEL_LEVEL ..
                make -j
                mv ../run/mini_app ../run/mini_app_hip_$PARELLEL_LEVEL
                ##add opt flags
                make clean
                cmake -DPARALLEL_LEVEL=$PARELLEL_LEVEL -DLDS=1 ..
                make -j
                mv ../run/mini_app ../run/mini_app_hip_${PARELLEL_LEVEL}_LDS

		        make clean
                cmake -DPARALLEL_LEVEL=$PARELLEL_LEVEL -DLDS=1 -DNO_TRANSPOSE=1 ..
		        make -j
                mv ../run/mini_app ../run/mini_app_hip_${PARELLEL_LEVEL}_LDS_NO_TRANSPOSE

		        make clean
                cmake -DPARALLEL_LEVEL=$PARELLEL_LEVEL -DLDS=1 -DNO_TRANSPOSE=1 -DFUSE_EG_C=1 ..
            	make -j
                mv ../run/mini_app ../run/mini_app_hip_${PARELLEL_LEVEL}_LDS_NO_TRANSPOSE_FUSE_EG_C
	        done
            cd ..
            cd ..
        fi
    fi
done
