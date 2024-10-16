#!/bin/bash
module load rocm craype-accel-amd-gfx90a 
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export options="-O3 -eZ -s real64 -homp -hfma -hfp3 -hunroll2 -I${MPICH_DIR}/include -L${MPICH_DIR}/lib -lmpi -L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_hsa"
