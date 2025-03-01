# Chemgen
This is a repository to generate chemistry for combustion solvers

## Testing
```bash
python -m pytest tests/
```

Options for testing:
- `--mech_file_ck`: Input mechanism file (Chemkin format)
- `--therm_file`: Thermodynamic data file 
- `--language`: Language for testing (fortran/python)
- `--temperature`: Temperature in Kelvin
- `--pressure`: Pressure in Pascal
- `--fuel`: Fuel composition
- `--oxidiser`: Oxidiser composition

## Usage

Generate GPU coefficients:
```bash
python generate_chemistry.py --mode gpu_coef \
    --mech mechanism.inp \
    --therm therm.dat \
    --output output_dir \
    --parallel-level 1
```
- `--parallel-level`: GPU parallelization level (1-4)

Generate getrates functions:
```bash 
python generate_chemistry.py --mode getrates \
    --mech mechanism.inp \
    --therm therm.dat \
    --output getrates.f90 \
    --language fortran \
    --omp
```

Key options:
- `--language`: Output language (fortran/python)
- `--omp`: Enable OpenMP support
- `--vector`: Generate vectorized version
- `--module`: Generate a fortran GPU module

Generate mini apps:
```bash
./generate_mini_apps.sh <path-to-chem>
# e.g.
./generate_mini_apps.sh ck_files/H2_burke
```
- resulting a folder named `mini_apps_<chem>`

Compile mini apps:
```bash
# request an interactive node for compilation
salloc --partition=gpu-dev --account=w47-gpu --time=01:30:00 --nodes=1 --gpus-per-node=8 --exclusive
# load the required modules and set enviroment variables
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
module load rocm craype-accel-amd-gfx90a
module load cmake/3.27.7
```
to compile all miniapps for a mechanism, run:
```bash
./compile_mini_apps.sh mini_apps_<chem>
# e.g.
./compile_mini_apps.sh mini_apps_H2_burke  
```


For a mechanism, run all mini apps to collect performance data:
```bash
./compare_mini_apps_write_csv.sh mini_apps_<chem>
#e.g.
./compare_mini_apps_write_csv.sh mini_apps_H2_burke
```

With the performance data in csv file. use a python script to visualise the comparison.


