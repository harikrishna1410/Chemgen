from .chemistry import chemistry
from pint import UnitRegistry
import jinja2

# Template for allocating and copying global arrays
allocate_and_copy_global_array = jinja2.Template("""
        err = hipMalloc(&{{name}}_d, sizeof({{dtype}}) * {{size}});
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate {{name}}_d: %s\\n", hipGetErrorString(err));
            return;
        }
        err = hipMemcpy({{name}}_d, {{name}}_h, sizeof({{dtype}}) * {{size}}, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to copy {{name}}_d: %s\\n", hipGetErrorString(err));
            return;
        }""")

# Template for copying constant arrays  
copy_constant_array = jinja2.Template("""
        err = hipMemcpyToSymbol({{name}}_d, {{name}}_h, sizeof({{dtype}}) * {{size}});
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to copy {{name}}_d: %s\\n", hipGetErrorString(err));
            return;
        }""")

# Template for allocating global arrays
alloc_global_array = jinja2.Template("""
        err = hipMalloc((void**)&{{name}}_d, {{size}} * sizeof({{dtype}}));
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate device memory for {{name}}_d: %s\\n", hipGetErrorString(err));
            return;
        }""")

# Universal constants
RU = 8.31451e7  # universal gas constant
SMALL = 1.0e-200  # a small value
PATM = 1.01325e6  # atmospheric pressure

# Default parallelization constants
DEFAULT_NREACT_PER_BLOCK = 1
DEFAULT_NSP_PER_BLOCK = 1
DEFAULT_VECLEN=64

# Element weights
ELEM_WT = {
    "H": 1.007969975471497E0,
    "O": 1.599940013885498E1,
    "C": 1.201115036010742E1,
    "N": 1.400669956207276E1
}


def get_constant_declarations(chem):
    """Generate constant declarations based on existing reaction types"""
    declarations = [
        "// Declare mechanism constants in constant memory"
    ]
    
    # Add arrays for each reaction type that exists
    if len(chem.get_reactions_by_type("standard")) > 0:
        declarations.extend([
            "__device__ __constant__ double A_d[NREACT_STD];",
            "__device__ __constant__ double B_d[NREACT_STD*2];",
            "__device__ __constant__ int sk_map_d[MAX_SP*2*NREACT_STD];",
            "__device__ __constant__ double sk_coef_d[MAX_SP*2*NREACT_STD];",
            "__device__ __constant__ int map_r_d[MAX_SP*NREACT_STD];",
            "__device__ __constant__ double coef_r_d[MAX_SP*NREACT_STD];",
            "__device__ __constant__ int map_p_d[MAX_SP*NREACT_STD];",
            "__device__ __constant__ double coef_p_d[MAX_SP*NREACT_STD];"
        ])
    if len(chem.get_reactions_by_type("troe")) > 0:
        declarations.extend([
            "__device__ __constant__ double A_0_troe_d[NREACT_TROE];",
            "__device__ __constant__ double B_0_troe_d[NREACT_TROE*2];",
            "__device__ __constant__ double A_inf_troe_d[NREACT_TROE];",
            "__device__ __constant__ double B_inf_troe_d[NREACT_TROE*2];",
            "__device__ __constant__ int sk_map_troe_d[MAX_SP*2*NREACT_TROE];",
            "__device__ __constant__ double sk_coef_troe_d[MAX_SP*2*NREACT_TROE];",
            "__device__ __constant__ int map_r_troe_d[MAX_SP*NREACT_TROE];",
            "__device__ __constant__ double coef_r_troe_d[MAX_SP*NREACT_TROE];",
            "__device__ __constant__ int map_p_troe_d[MAX_SP*NREACT_TROE];",
            "__device__ __constant__ double coef_p_troe_d[MAX_SP*NREACT_TROE];",
            "__device__ __constant__ double eff_fac_troe_d[NSP_RED*NREACT_TROE];",
            "__device__ __constant__ double fcent_coef_troe_d[6*NREACT_TROE];"
        ])
    if len(chem.get_reactions_by_type("third_body")) > 0:
        declarations.extend([
            "__device__ __constant__ double A_third_d[NREACT_THIRD];",
            "__device__ __constant__ double B_third_d[NREACT_THIRD*2];",
            "__device__ __constant__ int sk_map_third_d[MAX_SP*2*NREACT_THIRD];",
            "__device__ __constant__ double sk_coef_third_d[MAX_SP*2*NREACT_THIRD];",
            "__device__ __constant__ int map_r_third_d[MAX_SP*NREACT_THIRD];",
            "__device__ __constant__ double coef_r_third_d[MAX_SP*NREACT_THIRD];",
            "__device__ __constant__ int map_p_third_d[MAX_SP*NREACT_THIRD];",
            "__device__ __constant__ double coef_p_third_d[MAX_SP*NREACT_THIRD];",
            "__device__ __constant__ double eff_fac_third_d[NSP_RED*NREACT_THIRD];"
        ])
    if len(chem.get_reactions_by_type("plog")) > 0:
        declarations.extend([
            "__device__ __constant__ double A_plog_d[NREACT_PLOG];",
            "__device__ __constant__ double B_plog_d[NREACT_PLOG*2];",
            "__device__ __constant__ int sk_map_plog_d[MAX_SP*2*NREACT_PLOG];",
            "__device__ __constant__ double sk_coef_plog_d[MAX_SP*2*NREACT_PLOG];",
            "__device__ __constant__ int map_r_plog_d[MAX_SP*NREACT_PLOG];",
            "__device__ __constant__ double coef_r_plog_d[MAX_SP*NREACT_PLOG];",
            "__device__ __constant__ int map_p_plog_d[MAX_SP*NREACT_PLOG];",
            "__device__ __constant__ double coef_p_plog_d[MAX_SP*NREACT_PLOG];"
        ])
    
    # Add common arrays
    declarations.extend([
        "__device__ __constant__ double mw_d[NSP_RED];",
        "__device__ __constant__ double smh_coef_d[NSP_SK*14];",
        "__device__ __constant__ double T_mid_d[NSP_SK];"
    ])
    
    return "\n".join(declarations)

def get_copy_function(chem):
    """Generate copy function based on existing reaction types"""
    params = []
    copies = []
    
    # Add parameters and copies for each reaction type that exists
    if len(chem.get_reactions_by_type("standard")) > 0:
        params.extend([
            "const double* A_h,",
            "const double* B_h,",
            "const int* sk_map_h,",
            "const double* sk_coef_h,", 
            "const int* map_r_h,",
            "const double* coef_r_h,",
            "const int* map_p_h,",
            "const double* coef_p_h,"
        ])
        copies.extend([
            "hipError_t err;",
            copy_constant_array.render(name="A", size="NREACT_STD", dtype="double"),
            copy_constant_array.render(name="B", size="NREACT_STD*2", dtype="double"),
            copy_constant_array.render(name="sk_map", size="NREACT_STD*MAX_SP*2", dtype="int"),
            copy_constant_array.render(name="sk_coef", size="NREACT_STD*MAX_SP*2", dtype="double"),
            copy_constant_array.render(name="map_r", size="NREACT_STD*MAX_SP", dtype="int"),
            copy_constant_array.render(name="coef_r", size="NREACT_STD*MAX_SP", dtype="double"),
            copy_constant_array.render(name="map_p", size="NREACT_STD*MAX_SP", dtype="int"),
            copy_constant_array.render(name="coef_p", size="NREACT_STD*MAX_SP", dtype="double")
        ])
    if len(chem.get_reactions_by_type("troe")) > 0:
        params.extend([
            "const double* A_0_troe_h,",
            "const double* B_0_troe_h,",
            "const double* A_inf_troe_h,",
            "const double* B_inf_troe_h,",
            "const int* sk_map_troe_h,",
            "const double* sk_coef_troe_h,",
            "const int* map_r_troe_h,",
            "const double* coef_r_troe_h,",
            "const int* map_p_troe_h,",
            "const double* coef_p_troe_h,",
            "const double* eff_fac_troe_h,",
            "const double* fcent_coef_troe_h,"
        ])
        copies.extend([
            copy_constant_array.render(name="A_0_troe", size="NREACT_TROE", dtype="double"),
            copy_constant_array.render(name="B_0_troe", size="NREACT_TROE*2", dtype="double"),
            copy_constant_array.render(name="A_inf_troe", size="NREACT_TROE", dtype="double"),
            copy_constant_array.render(name="B_inf_troe", size="NREACT_TROE*2", dtype="double"),
            copy_constant_array.render(name="sk_map_troe", size="NREACT_TROE*MAX_SP*2", dtype="int"),
            copy_constant_array.render(name="sk_coef_troe", size="NREACT_TROE*MAX_SP*2", dtype="double"),
            copy_constant_array.render(name="map_r_troe", size="NREACT_TROE*MAX_SP", dtype="int"),
            copy_constant_array.render(name="coef_r_troe", size="NREACT_TROE*MAX_SP", dtype="double"),
            copy_constant_array.render(name="map_p_troe", size="NREACT_TROE*MAX_SP", dtype="int"),
            copy_constant_array.render(name="coef_p_troe", size="NREACT_TROE*MAX_SP", dtype="double"),
            copy_constant_array.render(name="eff_fac_troe", size="NSP_RED*NREACT_TROE", dtype="double"),
            copy_constant_array.render(name="fcent_coef_troe", size="6*NREACT_TROE", dtype="double")
        ])
    if len(chem.get_reactions_by_type("third_body")) > 0:
        params.extend([
            "const double* A_third_h,",
            "const double* B_third_h,",
            "const int* sk_map_third_h,",
            "const double* sk_coef_third_h,",
            "const int* map_r_third_h,",
            "const double* coef_r_third_h,",
            "const int* map_p_third_h,",
            "const double* coef_p_third_h,",
            "const double* eff_fac_third_h,"
        ])
        copies.extend([
            copy_constant_array.render(name="A_third", size="NREACT_THIRD", dtype="double"),
            copy_constant_array.render(name="B_third", size="NREACT_THIRD*2", dtype="double"),
            copy_constant_array.render(name="sk_map_third", size="NREACT_THIRD*MAX_SP*2", dtype="int"),
            copy_constant_array.render(name="sk_coef_third", size="NREACT_THIRD*MAX_SP*2", dtype="double"),
            copy_constant_array.render(name="map_r_third", size="NREACT_THIRD*MAX_SP", dtype="int"),
            copy_constant_array.render(name="coef_r_third", size="NREACT_THIRD*MAX_SP", dtype="double"),
            copy_constant_array.render(name="map_p_third", size="NREACT_THIRD*MAX_SP", dtype="int"),
            copy_constant_array.render(name="coef_p_third", size="NREACT_THIRD*MAX_SP", dtype="double"),
            copy_constant_array.render(name="eff_fac_third", size="NSP_RED*NREACT_THIRD", dtype="double")
        ])
    if len(chem.get_reactions_by_type("plog")) > 0:
        params.extend([
            "const double* A_plog_h,",
            "const double* B_plog_h,",
            "const int* sk_map_plog_h,",
            "const double* sk_coef_plog_h,",
            "const int* map_r_plog_h,",
            "const double* coef_r_plog_h,",
            "const int* map_p_plog_h,",
            "const double* coef_p_plog_h,"
        ])
        copies.extend([
            copy_constant_array.render(name="A_plog", size="NREACT_PLOG", dtype="double"),
            copy_constant_array.render(name="B_plog", size="NREACT_PLOG*2", dtype="double"),
            copy_constant_array.render(name="sk_map_plog", size="NREACT_PLOG*MAX_SP*2", dtype="int"),
            copy_constant_array.render(name="sk_coef_plog", size="NREACT_PLOG*MAX_SP*2", dtype="double"),
            copy_constant_array.render(name="map_r_plog", size="NREACT_PLOG*MAX_SP", dtype="int"),
            copy_constant_array.render(name="coef_r_plog", size="NREACT_PLOG*MAX_SP", dtype="double"),
            copy_constant_array.render(name="map_p_plog", size="NREACT_PLOG*MAX_SP", dtype="int"),
            copy_constant_array.render(name="coef_p_plog", size="NREACT_PLOG*MAX_SP", dtype="double")
        ])
    
    # Add common parameters and copies
    params.extend([
        "const double* mw_h,",
        "const double* smh_coef_h,",
        "const double* T_mid_h"
    ])
    copies.extend([
        copy_constant_array.render(name="mw", size="NSP_RED", dtype="double"),
        copy_constant_array.render(name="smh_coef", size="NSP_SK*14", dtype="double"),
        copy_constant_array.render(name="T_mid", size="NSP_SK", dtype="double")
    ])
    
    function = f"""extern "C" {{
    void copyConstantsToDevice({(chr(10) + " "*27).join(params)})
    {{
        {(chr(10) + " "*8).join(copies)}
    }}
}}"""
    
    return function

def get_constant_declarations_rocblas(chem):
    """Generate constant declarations based on existing reaction types"""
    declarations = [
        "// Declare mechanism constants in constant memory",
        "//Using global memory here because rocBLAS doesn't seem to work with constant memory"
    ]
    
    # Add arrays for each reaction type that exists
    if len(chem.get_reactions_by_type("standard")) > 0:
        declarations.extend([
            "__device__ double *A_d;",
            "__device__ double *B_d;",
            "__device__ double *sk_coef_d;",
            "__device__ double *coef_r_d;",
            "__device__ double *coef_p_d;",
            "__device__ double *wdot_coef_d;"
        ])
    if len(chem.get_reactions_by_type("troe")) > 0:
        declarations.extend([
            "__device__ double *A_0_troe_d;",
            "__device__ double *B_0_troe_d;",
            "__device__ double *A_inf_troe_d;",
            "__device__ double *B_inf_troe_d;",
            "__device__ double *sk_coef_troe_d;",
            "__device__ double *coef_r_troe_d;",
            "__device__ double *coef_p_troe_d;",
            "__device__ double *eff_fac_troe_d;",
            "__device__ double *fcent_coef_troe_d;",
            "__device__ double *wdot_coef_troe_d;"
        ])
    if len(chem.get_reactions_by_type("third_body")) > 0:
        declarations.extend([
            "__device__ double *A_third_d;",
            "__device__ double *B_third_d;",
            "__device__ double *sk_coef_third_d;",
            "__device__ double *coef_r_third_d;",
            "__device__ double *coef_p_third_d;",
            "__device__ double *eff_fac_third_d;",
            "__device__ double *wdot_coef_third_d;"
        ])
    if len(chem.get_reactions_by_type("plog")) > 0:
        declarations.extend([
            "__device__ double *A_plog_d;",
            "__device__ double *B_plog_d;",
            "__device__ double *sk_coef_plog_d;",
            "__device__ double *coef_r_plog_d;",
            "__device__ double *coef_p_plog_d;"
            "__device__ double *wdot_coef_plog_d;"
        ])
    # Add common arrays
    declarations.extend([
        "__device__ __constant__ double mw_d[NSP_RED];",
        "__device__ __constant__ double smh_coef_d[NSP_SK*14];",
        "__device__ __constant__ double T_mid_d[NSP_SK];",
        "//intermediate arrays",
        "__device__ double *rr_d;",
        "__device__ double *sigma_logC_r_d;",
        "__device__ double *sigma_logC_p_d;",
        "__device__ double *logEQK_d;"
    ])

    if len(chem.get_reactions_by_type("troe")) > 0:
        declarations.extend([
        "__device__ double *rr_troe_d;",
        "__device__ double *sigma_logC_r_troe_d;",
        "__device__ double *sigma_logC_p_troe_d;",
        "__device__ double *logEQK_troe_d;"
        ])
    if len(chem.get_reactions_by_type("third_body")) > 0:
        declarations.extend([
        "__device__ double *rr_third_d;",
        "__device__ double *sigma_logC_r_third_d;",
        "__device__ double *sigma_logC_p_third_d;",
        "__device__ double *logEQK_third_d;"
        ])
    if len(chem.get_reactions_by_type("plog")) > 0:
        declarations.extend([
        "__device__ double *rr_plog_d;",
        "__device__ double *sigma_logC_r_plog_d;",
        "__device__ double *sigma_logC_p_plog_d;",
        "__device__ double *logEQK_plog_d;"
        ])
    
    return "\n".join(declarations)

def get_copy_function_rocblas(chem):
    """Generate copy function based on existing reaction types"""
    params = []
    copies = []
    
    # Add parameters and copies for each reaction type that exists
    if len(chem.get_reactions_by_type("standard")) > 0:
        params.extend([
            "const double* A_h,",
            "const double* B_h,", 
            "const double* sk_coef_h,",
            "const double* coef_r_h,",
            "const double* coef_p_h,",
            "const double* wdot_coef_h,"
        ])
        copies.extend([
            "hipError_t err;",
            allocate_and_copy_global_array.render(name="A", size="NREACT_STD", dtype="double"),
            allocate_and_copy_global_array.render(name="B", size="NREACT_STD * 2", dtype="double"),
            allocate_and_copy_global_array.render(name="sk_coef", size="NREACT_STD * NSP_SK", dtype="double"),
            allocate_and_copy_global_array.render(name="coef_r", size="NREACT_STD * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="coef_p", size="NREACT_STD * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="wdot_coef", size="NREACT_STD * NSP_RED", dtype="double")
        ])

    if len(chem.get_reactions_by_type("troe")) > 0:
        params.extend([
            "const double* A_0_troe_h,",
            "const double* B_0_troe_h,",
            "const double* A_inf_troe_h,", 
            "const double* B_inf_troe_h,",
            "const double* sk_coef_troe_h,",
            "const double* coef_r_troe_h,",
            "const double* coef_p_troe_h,",
            "const double* wdot_coef_troe_h,",
            "const double* eff_fac_troe_h,",
            "const double* fcent_coef_troe_h,"
        ])
        copies.extend([
            allocate_and_copy_global_array.render(name="A_0_troe", size="NREACT_TROE", dtype="double"),
            allocate_and_copy_global_array.render(name="B_0_troe", size="NREACT_TROE * 2", dtype="double"),
            allocate_and_copy_global_array.render(name="A_inf_troe", size="NREACT_TROE", dtype="double"),
            allocate_and_copy_global_array.render(name="B_inf_troe", size="NREACT_TROE * 2", dtype="double"),
            allocate_and_copy_global_array.render(name="sk_coef_troe", size="NREACT_TROE * NSP_SK", dtype="double"),
            allocate_and_copy_global_array.render(name="coef_r_troe", size="NREACT_TROE * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="coef_p_troe", size="NREACT_TROE * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="wdot_coef_troe", size="NREACT_TROE * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="eff_fac_troe", size="NSP_RED * NREACT_TROE", dtype="double"),
            allocate_and_copy_global_array.render(name="fcent_coef_troe", size="6 * NREACT_TROE", dtype="double")
        ])

    if len(chem.get_reactions_by_type("third_body")) > 0:
        params.extend([
            "const double* A_third_h,",
            "const double* B_third_h,",
            "const double* sk_coef_third_h,",
            "const double* coef_r_third_h,",
            "const double* coef_p_third_h,",
            "const double* wdot_coef_third_h,",
            "const double* eff_fac_third_h,"
        ])
        copies.extend([
            allocate_and_copy_global_array.render(name="A_third", size="NREACT_THIRD", dtype="double"),
            allocate_and_copy_global_array.render(name="B_third", size="NREACT_THIRD * 2", dtype="double"),
            allocate_and_copy_global_array.render(name="sk_coef_third", size="NREACT_THIRD * NSP_SK", dtype="double"),
            allocate_and_copy_global_array.render(name="coef_r_third", size="NREACT_THIRD * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="coef_p_third", size="NREACT_THIRD * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="wdot_coef_third", size="NREACT_THIRD * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="eff_fac_third", size="NSP_RED * NREACT_THIRD", dtype="double")
        ])

    if len(chem.get_reactions_by_type("plog")) > 0:
        params.extend([
            "const double* A_plog_h,",
            "const double* B_plog_h,",
            "const double* sk_coef_plog_h,",
            "const double* coef_r_plog_h,",
            "const double* coef_p_plog_h,",
            "const double* wdot_coef_plog_h,",
        ])
        copies.extend([
            allocate_and_copy_global_array.render(name="A_plog", size="NREACT_PLOG", dtype="double"),
            allocate_and_copy_global_array.render(name="B_plog", size="NREACT_PLOG * 2", dtype="double"),
            allocate_and_copy_global_array.render(name="sk_coef_plog", size="NREACT_PLOG * NSP_SK", dtype="double"),
            allocate_and_copy_global_array.render(name="coef_r_plog", size="NREACT_PLOG * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="coef_p_plog", size="NREACT_PLOG * NSP_RED", dtype="double"),
            allocate_and_copy_global_array.render(name="wdot_coef_plog", size="NREACT_PLOG * NSP_RED", dtype="double"),
        ])
    
    # Add common parameters and copies
    params.extend([
        "const double* mw_h,",
        "const double* smh_coef_h,",
        "const double* T_mid_h"
    ])
    copies.extend([
        copy_constant_array.render(name="mw", size="NSP_RED", dtype="double"),
        copy_constant_array.render(name="smh_coef", size="NSP_SK * 14", dtype="double"),
        copy_constant_array.render(name="T_mid", size="NSP_SK", dtype="double")
    ])
    
    function = f"""extern "C" {{
    void copyConstantsToDevice({(chr(10) + " "*27).join(params)})
    {{
        {(chr(10)).join(copies)}
    }}
}}"""

    function += """
extern "C" {
    void allocate_intermediate_DeviceMemory(int ng) {
        hipError_t err;
"""
    # Add standard reaction allocations
    function += "\n" + alloc_global_array.render(name="rr", size="ng * NREACT_STD", dtype="double")
    function += "\n" + alloc_global_array.render(name="sigma_logC_r", size="ng * NREACT_STD", dtype="double")
    function += "\n" + alloc_global_array.render(name="sigma_logC_p", size="ng * NREACT_STD", dtype="double")
    function += "\n" + alloc_global_array.render(name="logEQK", size="ng * NREACT_STD", dtype="double")

    # Add third body reaction allocations if needed
    if len(chem.get_reactions_by_type("third_body")) > 0:
        function += "\n" + alloc_global_array.render(name="rr_third", size="ng * NREACT_THIRD", dtype="double")
        function += "\n" + alloc_global_array.render(name="sigma_logC_r_third", size="ng * NREACT_THIRD", dtype="double")
        function += "\n" + alloc_global_array.render(name="sigma_logC_p_third", size="ng * NREACT_THIRD", dtype="double")
        function += "\n" + alloc_global_array.render(name="logEQK_third", size="ng * NREACT_THIRD", dtype="double")

    # Add troe reaction allocations if needed
    if len(chem.get_reactions_by_type("troe")) > 0:
        function += "\n" + alloc_global_array.render(name="rr_troe", size="ng * NREACT_TROE", dtype="double")
        function += "\n" + alloc_global_array.render(name="sigma_logC_r_troe", size="ng * NREACT_TROE", dtype="double")
        function += "\n" + alloc_global_array.render(name="sigma_logC_p_troe", size="ng * NREACT_TROE", dtype="double")
        function += "\n" + alloc_global_array.render(name="logEQK_troe", size="ng * NREACT_TROE", dtype="double")

    # Add plog reaction allocations if needed
    if len(chem.get_reactions_by_type("plog")) > 0:
        function += "\n" + alloc_global_array.render(name="rr_plog", size="ng * NREACT_PLOG", dtype="double")
        function += "\n" + alloc_global_array.render(name="sigma_logC_r_plog", size="ng * NREACT_PLOG", dtype="double")
        function += "\n" + alloc_global_array.render(name="sigma_logC_p_plog", size="ng * NREACT_PLOG", dtype="double")
        function += "\n" + alloc_global_array.render(name="logEQK_plog", size="ng * NREACT_PLOG", dtype="double")

    function += """
        return;
    }
}"""
    
    return function

def get_header_content(chem: chemistry,
                       parallel_level=1,
                       veclen=DEFAULT_VECLEN,
                       nreact_per_block=DEFAULT_NREACT_PER_BLOCK,
                       rocblas=False):
    max_specs = chem.find_max_specs(parallel_level>2)
    header_content = f"""#ifndef CONSTANTS_V{parallel_level}_H
#define CONSTANTS_V{parallel_level}_H
#include "hip/hip_runtime.h"
///****************mechanism constants************************
//number of reduced species
#define NSP_RED {len(chem.reduced_species)}
//number of skeletal species
#define NSP_SK {len(chem.species)}
//number of qssa species
#define NSP_QSSA {len(chem.qssa_species)}
///these are total number of reactions in the mech
//these variables had to be divided by NREACT_PER_BLOCK
#define NREACT_MECH {len(chem.reactions)}
#define NREACT_STD {len(chem.get_reactions_by_type("standard"))}
#define NREACT_TROE {len(chem.get_reactions_by_type("troe"))}
#define NREACT_THIRD {len(chem.get_reactions_by_type("third_body"))}
#define NREACT_PLOG {len(chem.get_reactions_by_type("plog"))}
//
//maximum number of species involved in a reaction
//either on reactants side or products side
//MAX_SP should be divisible by NSPEC_PER_THREAD when using v3 parallelisation
#define MAX_SP {max_specs}
//twice of above
#define MAX_SP2 {max_specs*2}
//max third body third body reactants
#define MAX_THIRD_BODIES {chem.find_max_third_body()}
//veclen
#define VECLEN {veclen}
//when using more v2 and v3 parallelisation
//this had to divide all NREACT_* variables
#define NREACT_PER_BLOCK {nreact_per_block if nreact_per_block is not None else DEFAULT_NREACT_PER_BLOCK} //number of reactions solved per thread block
//this is used in v3 parallelisation
//MAX_SP should be divisible by NSP_PER_THREAD when using v3 parallelisation
//this is always some power of 2
#define NSP_PER_BLOCK {DEFAULT_NSP_PER_BLOCK}
//max number of species per thread block
//ratio of MAX_SP/NSP_PER_THREAD should be divisible by 2
#define SP_PER_THREAD {max_specs // DEFAULT_NSP_PER_BLOCK}
//twice of above
#define SP2_PER_THREAD {(max_specs * 2) // DEFAULT_NSP_PER_BLOCK}

const double RU = {RU}; //universal gas constant
const double SMALL = {SMALL}; //a small value
const double PATM = {PATM}; //atmospheric pressure
///****************mechanism constants************************
{get_constant_declarations(chem) if not rocblas else get_constant_declarations_rocblas(chem)}

{get_copy_function(chem) if not rocblas else get_copy_function_rocblas(chem)}
#endif
"""
    return header_content