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
