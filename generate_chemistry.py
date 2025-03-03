import argparse
import os
from chemgen.parser import ckparser
from chemgen.chemistry import chemistry, chemistry_expressions
from chemgen.write_constants import write_coef_module, write_constants_header

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate GPU coefficients and getrates functions')
    
    # Main operation choice
    parser.add_argument('--mode', choices=['gpu_coef', 'getrates', "mini_app"], required=True,
                      help='Mode of operation: generate GPU coefficients or getrates function')
    
    # Input files
    parser.add_argument('--mech', help='Path to mechanism input file')
    parser.add_argument('--therm', help='Path to thermodynamic data file')
    parser.add_argument('--yaml_file', help='Path to YAML mechanism file')
    parser.add_argument('--output', required=True, help='Output file or directory')
    
    # GPU coefficient options
    parser.add_argument('--parallel-level', type=int, choices=[1,2,3,4], default=1,
                      help='Parallelization level for GPU coefficients')
    parser.add_argument('--veclen', type=int, default=16,
                      help='Vector length for GPU coefficients')
    parser.add_argument('--nreact-per-block', type=int, default=8,
                      help='Number of reactions per block')
    
    # Getrates options
    parser.add_argument('--language', choices=['python', 'fortran'], default='fortran',
                      help='Output language for getrates')
    parser.add_argument('--omp', action='store_true',
                      help='Enable OpenMP support')
    parser.add_argument('--vector', action='store_true',
                      help='Enable vectorized version')
    parser.add_argument('--module', action='store_true',
                      help='Generate as module (for Fortran)')
    parser.add_argument('--input-mw', action='store_true',
                      help='Take molecular weights as input')
    parser.add_argument('--rtypes-together', action='store_true',
                      help='Write reaction types together')
    
    #mini app options
    parser.add_argument('--ng', type=int, default=1,
                      help='Number of grid points')
    parser.add_argument('--ncpu', type=int, default=64,
                        help='Number of CPUs per node')
    parser.add_argument('--ngpu', type=int, default=8,
                        help='Number of GPUs per node')
    parser.add_argument('--nt', type=int, default=1,
                        help='Number of time steps')
    parser.add_argument('--time-cpu', action='store_true',
                        help='Time the CPU code')

    return parser.parse_args()

def generate_gpu_coefficients(args):
    # Create parser and chemistry objects
    ckp = ckparser()
    chem = chemistry(args.mech, ckp, therm_file=args.therm, yaml_file=args.yaml_file)
    
    # Generate coefficient module
    write_coef_module(args.output, chem, 
                     parallel_level=args.parallel_level,
                     nreact_per_block=args.nreact_per_block)
    
    # Generate constants header
    write_constants_header(args.output, chem,
                         parallel_level=args.parallel_level,
                         veclen=args.veclen,
                         nreact_per_block=args.nreact_per_block)

def generate_getrates(args):
    # Create parser and chemistry objects
    ckp = ckparser()
    chem = chemistry(args.mech, ckp, therm_file=args.therm, yaml_file=args.yaml_file)
    
    # Create chemistry expressions object with specified options
    chem_expr = chemistry_expressions(chem,
                                   vec=args.vector,
                                   omp=args.omp,
                                   mod=args.module,
                                   language=args.language)
    
    # Write expressions to file
    chem_expr.write_expressions_to_file(args.output,
                                      write_rtypes_together=args.rtypes_together,
                                      input_MW=args.input_mw)

def generate_mini_app(args):
    # Create parser and chemistry objects
    ckp = ckparser()
    chem = chemistry(args.mech, ckp, therm_file=args.therm, yaml_file=args.yaml_file)
    

    src_dir = os.path.join(args.output,f"src")
    os.makedirs(src_dir,exist_ok=True)
    #write scalar f90
    # Create chemistry expressions object with specified options
    filename = os.path.join(src_dir,f"getrates.f90")
    chem_expr = chemistry_expressions(chem,
                                   vec=False,
                                   omp=False,
                                   mod=False,
                                   language=args.language)
    
    # Write expressions to file
    chem_expr.write_expressions_to_file(filename,
                                      write_rtypes_together=args.rtypes_together,
                                      input_MW=args.input_mw)
    
    #write vector f90
    # Create chemistry expressions object with specified options
    filename = os.path.join(src_dir,f"getrates_i.f90")
    chem_expr = chemistry_expressions(chem,
                                   vec=True,
                                   omp=False,
                                   mod=False,
                                   language=args.language)
    
    # Write expressions to file
    chem_expr.write_expressions_to_file(filename,
                                      write_rtypes_together=args.rtypes_together,
                                      input_MW=args.input_mw)
    
    #write chemgen f90
    # Create chemistry expressions object with specified options
    filename = os.path.join(src_dir,f"chemgen_m.f90")
    chem_expr = chemistry_expressions(chem,
                                   vec=False,
                                   omp=True,
                                   mod=True,
                                   language=args.language)
    
    # Write expressions to file
    chem_expr.write_expressions_to_file(filename,
                                      write_rtypes_together=args.rtypes_together,
                                      input_MW=args.input_mw)

    inc_dir = os.path.join(src_dir,"include")
    mod_dir = os.path.join(src_dir,"modules")
    os.makedirs(inc_dir, exist_ok=True)
    os.makedirs(mod_dir, exist_ok=True)
    for parallel_level in [1,2,3,4]:
        # Generate coefficient module
        write_coef_module(mod_dir, chem, 
                         parallel_level=parallel_level,
                         nreact_per_block=args.nreact_per_block)
        # Generate constants header
        write_constants_header(inc_dir, chem,
                             parallel_level=parallel_level,
                             veclen=args.veclen,
                             nreact_per_block=args.nreact_per_block)
    
    #write hip kernels
    chem_expr.write_hip_kernels(src_dir)
    #write mini app
    chem_expr.write_chemistry_mini_app(src_dir,
                                       args.ng,
                                       ncpu=args.ncpu,
                                       ngpu=args.ngpu,
                                       nt=args.nt,
                                       input_MW=args.input_mw,
                                       time_cpu=args.time_cpu)

def main():
    args = parse_arguments()
    
    if args.mode == 'gpu_coef':
        generate_gpu_coefficients(args)
    elif args.mode == "mini_app":
        generate_mini_app(args)
    else:  # getrates
        generate_getrates(args)

if __name__ == "__main__":
    main()