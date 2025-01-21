import argparse
import os
from chemgen.parser import ckparser
from chemgen.chemistry import chemistry, chemistry_expressions
from chemgen.write_constants import write_coef_module, write_constants_header

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate GPU coefficients and getrates functions')
    
    # Main operation choice
    parser.add_argument('--mode', choices=['gpu_coef', 'getrates'], required=True,
                      help='Mode of operation: generate GPU coefficients or getrates function')
    
    # Input files
    parser.add_argument('--mech', required=True, help='Path to mechanism input file')
    parser.add_argument('--therm', required=True, help='Path to thermodynamic data file')
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

    return parser.parse_args()

def generate_gpu_coefficients(args):
    # Create parser and chemistry objects
    ckp = ckparser()
    chem = chemistry(args.mech, ckp, therm_file=args.therm)
    
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    
    # Generate coefficient module
    write_coef_module(output_dir, chem, 
                     parallel_level=args.parallel_level,
                     nreact_per_block=args.nreact_per_block)
    
    # Generate constants header
    write_constants_header(output_dir, chem,
                         parallel_level=args.parallel_level,
                         veclen=args.veclen,
                         nreact_per_block=args.nreact_per_block)

def generate_getrates(args):
    # Create parser and chemistry objects
    ckp = ckparser()
    chem = chemistry(args.mech, ckp, therm_file=args.therm)
    
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

def main():
    args = parse_arguments()
    
    if args.mode == 'gpu_coef':
        generate_gpu_coefficients(args)
    else:  # getrates
        generate_getrates(args)

if __name__ == "__main__":
    main()