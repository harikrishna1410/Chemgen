from .chemistry import chemistry
from pint import UnitRegistry

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
            "err = hipMemcpyToSymbol(A_d, A_h, sizeof(double) * NREACT_STD);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(B_d, B_h, sizeof(double) * NREACT_STD * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(sk_map_d, sk_map_h, sizeof(int) * NREACT_STD * MAX_SP * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_map_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(sk_coef_d, sk_coef_h, sizeof(double) * NREACT_STD * MAX_SP * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_coef_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(map_r_d, map_r_h, sizeof(int) * NREACT_STD * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy map_r_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(coef_r_d, coef_r_h, sizeof(double) * NREACT_STD * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_r_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(map_p_d, map_p_h, sizeof(int) * NREACT_STD * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy map_p_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(coef_p_d, coef_p_h, sizeof(double) * NREACT_STD * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_p_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}"
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
            "const double* fcent_coef_troe_h,",
        ])
        copies.extend([
            "err = hipMemcpyToSymbol(A_0_troe_d, A_0_troe_h, sizeof(double) * NREACT_TROE);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(B_0_troe_d, B_0_troe_h, sizeof(double) * NREACT_TROE * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(A_inf_troe_d, A_inf_troe_h, sizeof(double) * NREACT_TROE);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(B_inf_troe_d, B_inf_troe_h, sizeof(double) * NREACT_TROE * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(sk_map_troe_d, sk_map_troe_h, sizeof(int) * NREACT_TROE * MAX_SP * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_map_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(sk_coef_troe_d, sk_coef_troe_h, sizeof(double) * NREACT_TROE * MAX_SP * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_coef_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(map_r_troe_d, map_r_troe_h, sizeof(int) * NREACT_TROE * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy map_r_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(coef_r_troe_d, coef_r_troe_h, sizeof(double) * NREACT_TROE * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_r_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(map_p_troe_d, map_p_troe_h, sizeof(int) * NREACT_TROE * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy map_p_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(coef_p_troe_d, coef_p_troe_h, sizeof(double) * NREACT_TROE * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_p_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(eff_fac_troe_d, eff_fac_troe_h, sizeof(double) * NSP_RED * NREACT_TROE);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy eff_fac_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(fcent_coef_troe_d, fcent_coef_troe_h, sizeof(double) * 6 * NREACT_TROE);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy eff_fac_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}"
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
            "err = hipMemcpyToSymbol(A_third_d, A_third_h, sizeof(double) * NREACT_THIRD);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(B_third_d, B_third_h, sizeof(double) * NREACT_THIRD * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(sk_map_third_d, sk_map_third_h, sizeof(int) * NREACT_THIRD * MAX_SP * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_map_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(sk_coef_third_d, sk_coef_third_h, sizeof(double) * NREACT_THIRD * MAX_SP * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_coef_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(map_r_third_d, map_r_third_h, sizeof(int) * NREACT_THIRD * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy map_r_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(coef_r_third_d, coef_r_third_h, sizeof(double) * NREACT_THIRD * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_r_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(map_p_third_d, map_p_third_h, sizeof(int) * NREACT_THIRD * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy map_p_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(coef_p_third_d, coef_p_third_h, sizeof(double) * NREACT_THIRD * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_p_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(eff_fac_third_d, eff_fac_third_h, sizeof(double) * NSP_RED * NREACT_THIRD);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy eff_fac_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}"
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
            "err = hipMemcpyToSymbol(A_plog_d, A_plog_h, sizeof(double) * NREACT_PLOG);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(B_plog_d, B_plog_h, sizeof(double) * NREACT_PLOG * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(sk_map_plog_d, sk_map_plog_h, sizeof(int) * NREACT_PLOG * MAX_SP * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_map_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(sk_coef_plog_d, sk_coef_plog_h, sizeof(double) * NREACT_PLOG * MAX_SP * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_coef_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(map_r_plog_d, map_r_plog_h, sizeof(int) * NREACT_PLOG * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy map_r_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(coef_r_plog_d, coef_r_plog_h, sizeof(double) * NREACT_PLOG * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_r_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(map_p_plog_d, map_p_plog_h, sizeof(int) * NREACT_PLOG * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy map_p_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpyToSymbol(coef_p_plog_d, coef_p_plog_h, sizeof(double) * NREACT_PLOG * MAX_SP);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_p_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}"
        ])
    
    # Add common parameters and copies
    params.extend([
        "const double* mw_h,",
        "const double* smh_coef_h,",
        "const double* T_mid_h"
    ])
    copies.extend([
        "err = hipMemcpyToSymbol(mw_d, mw_h, sizeof(double) * NSP_RED);",
        "if (err != hipSuccess) {",
        "    fprintf(stderr, \"Failed to copy mw_d: %s\\n\", hipGetErrorString(err));",
        "    return;",
        "}",
        "err = hipMemcpyToSymbol(smh_coef_d, smh_coef_h, sizeof(double) * NSP_SK * 14);",
        "if (err != hipSuccess) {",
        "    fprintf(stderr, \"Failed to copy smh_coef_d: %s\\n\", hipGetErrorString(err));",
        "    return;",
        "}",
        "err = hipMemcpyToSymbol(T_mid_d, T_mid_h, sizeof(double) * NSP_SK);",
        "if (err != hipSuccess) {",
        "    fprintf(stderr, \"Failed to copy T_mid_d: %s\\n\", hipGetErrorString(err));",
        "    return;",
        "}"
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
        "// Declare mechanism constants in constant memory"
    ]
    
    # Add arrays for each reaction type that exists
    if len(chem.get_reactions_by_type("standard")) > 0:
        declarations.extend([
            "__device__ double *A_d;",
            "__device__ double *B_d;",
            "__device__ double *sk_coef_d;",
            "__device__ double *coef_r_d;",
            "__device__ double *coef_p_d;"
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
            "__device__ double *fcent_coef_troe_d;"
        ])
    if len(chem.get_reactions_by_type("third_body")) > 0:
        declarations.extend([
            "__device__ double *A_third_d;",
            "__device__ double *B_third_d;",
            "__device__ double *sk_coef_third_d;",
            "__device__ double *coef_r_third_d;",
            "__device__ double *coef_p_third_d;",
            "__device__ double *eff_fac_third_d;"
        ])
    if len(chem.get_reactions_by_type("plog")) > 0:
        declarations.extend([
            "__device__ double *A_plog_d;",
            "__device__ double *B_plog_d;",
            "__device__ double *sk_coef_plog_d;",
            "__device__ double *coef_r_plog_d;",
            "__device__ double *coef_p_plog_d;"
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
            "const double* coef_p_h,"
        ])
        copies.extend([
            "hipError_t err;",
            "err = hipMalloc(&A_d, sizeof(double) * NREACT_STD);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate A_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&B_d, sizeof(double) * NREACT_STD * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate B_d: %s\\n\", hipGetErrorString(err));", 
            "    return;",
            "}",
            "err = hipMalloc(&sk_coef_d, sizeof(double) * NREACT_STD * NSP_SK);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate sk_coef_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&coef_r_d, sizeof(double) * NREACT_STD * NSP_RED);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate coef_r_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&coef_p_d, sizeof(double) * NREACT_STD * NSP_RED);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate coef_p_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(A_d, A_h, sizeof(double) * NREACT_STD, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(B_d, B_h, sizeof(double) * NREACT_STD * 2, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(sk_coef_d, sk_coef_h, sizeof(double) * NREACT_STD * NSP_SK, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_coef_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(coef_r_d, coef_r_h, sizeof(double) * NREACT_STD * NSP_RED, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_r_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(coef_p_d, coef_p_h, sizeof(double) * NREACT_STD * NSP_RED, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_p_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}"
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
            "const double* eff_fac_troe_h,",
            "const double* fcent_coef_troe_h,"
        ])
        copies.extend([
            "err = hipMalloc(&A_0_troe_d, sizeof(double) * NREACT_TROE);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate A_0_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&B_0_troe_d, sizeof(double) * NREACT_TROE * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate B_0_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&A_inf_troe_d, sizeof(double) * NREACT_TROE);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate A_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&B_inf_troe_d, sizeof(double) * NREACT_TROE * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate B_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&sk_coef_troe_d, sizeof(double) * NREACT_TROE * NSP_SK);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate sk_coef_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&coef_r_troe_d, sizeof(double) * NREACT_TROE * NSP_RED);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate coef_r_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&coef_p_troe_d, sizeof(double) * NREACT_TROE * NSP_RED);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate coef_p_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&eff_fac_troe_d, sizeof(double) * NSP_RED * NREACT_TROE);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate eff_fac_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&fcent_coef_troe_d, sizeof(double) * 6 * NREACT_TROE);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate eff_fac_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(A_troe_d, A_troe_h, sizeof(double) * NREACT_TROE, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(B_troe_d, B_troe_h, sizeof(double) * NREACT_TROE * 2, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(sk_coef_troe_d, sk_coef_troe_h, sizeof(double) * NREACT_TROE * NSP_SK, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_coef_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(coef_r_troe_d, coef_r_troe_h, sizeof(double) * NREACT_TROE * NSP_RED, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_r_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(coef_p_troe_d, coef_p_troe_h, sizeof(double) * NREACT_TROE * NSP_RED, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_p_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(eff_fac_troe_d, eff_fac_troe_h, sizeof(double) * NSP_RED * NREACT_TROE, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy eff_fac_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(fcent_coef_troe_d, fcent_coef_troe_h, sizeof(double) * 6 * NREACT_TROE, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy fcent_coef_troe_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}"
        ])
    if len(chem.get_reactions_by_type("third_body")) > 0:
        params.extend([
            "const double* A_third_h,",
            "const double* B_third_h,",
            "const double* sk_coef_third_h,",
            "const double* coef_r_third_h,",
            "const double* coef_p_third_h,",
            "const double* eff_fac_third_h,"
        ])
        copies.extend([
            "err = hipMalloc(&A_third_d, sizeof(double) * NREACT_THIRD);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate A_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&B_third_d, sizeof(double) * NREACT_THIRD * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate B_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&sk_coef_third_d, sizeof(double) * NREACT_THIRD * NSP_SK);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate sk_coef_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&coef_r_third_d, sizeof(double) * NREACT_THIRD * NSP_RED);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate coef_r_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&coef_p_third_d, sizeof(double) * NREACT_THIRD * NSP_RED);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate coef_p_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&eff_fac_third_d, sizeof(double) * NSP_RED * NREACT_THIRD);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate eff_fac_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(A_third_d, A_third_h, sizeof(double) * NREACT_THIRD, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(B_third_d, B_third_h, sizeof(double) * NREACT_THIRD * 2, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(sk_coef_third_d, sk_coef_third_h, sizeof(double) * NREACT_THIRD * NSP_SK, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_coef_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(coef_r_third_d, coef_r_third_h, sizeof(double) * NREACT_THIRD * NSP_RED, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_r_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(coef_p_third_d, coef_p_third_h, sizeof(double) * NREACT_THIRD * NSP_RED, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_p_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(eff_fac_third_d, eff_fac_third_h, sizeof(double) * NSP_RED * NREACT_THIRD, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy eff_fac_third_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}"
        ])
    if len(chem.get_reactions_by_type("plog")) > 0:
        params.extend([
            "const double* A_plog_h,",
            "const double* B_plog_h,",
            "const double* sk_coef_plog_h,",
            "const double* coef_r_plog_h,",
            "const double* coef_p_plog_h,"
        ])
        copies.extend([
            "err = hipMalloc(&A_plog_d, sizeof(double) * NREACT_PLOG);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate A_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&B_plog_d, sizeof(double) * NREACT_PLOG * 2);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate B_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&sk_coef_plog_d, sizeof(double) * NREACT_PLOG * NSP_SK);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate sk_coef_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&coef_r_plog_d, sizeof(double) * NREACT_PLOG * NSP_RED);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate coef_r_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMalloc(&coef_p_plog_d, sizeof(double) * NREACT_PLOG * NSP_RED);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to allocate coef_p_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(A_plog_d, A_plog_h, sizeof(double) * NREACT_PLOG, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy A_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(B_plog_d, B_plog_h, sizeof(double) * NREACT_PLOG * 2, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy B_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(sk_coef_plog_d, sk_coef_plog_h, sizeof(double) * NREACT_PLOG * NSP_SK, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy sk_coef_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(coef_r_plog_d, coef_r_plog_h, sizeof(double) * NREACT_PLOG * NSP_RED, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_r_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}",
            "err = hipMemcpy(coef_p_plog_d, coef_p_plog_h, sizeof(double) * NREACT_PLOG * NSP_RED, hipMemcpyHostToDevice);",
            "if (err != hipSuccess) {",
            "    fprintf(stderr, \"Failed to copy coef_p_plog_d: %s\\n\", hipGetErrorString(err));",
            "    return;",
            "}"
        ])
    
    # Add common parameters and copies
    params.extend([
        "const double* mw_h,",
        "const double* smh_coef_h,",
        "const double* T_mid_h"
    ])
    copies.extend([
        "err = hipMemcpyToSymbol(mw_d, mw_h, sizeof(double) * NSP_RED);",
        "if (err != hipSuccess) {",
        "    fprintf(stderr, \"Failed to copy mw_d: %s\\n\", hipGetErrorString(err));", 
        "    return;",
        "}",
        "err = hipMemcpyToSymbol(smh_coef_d, smh_coef_h, sizeof(double) * NSP_SK * 14);",
        "if (err != hipSuccess) {",
        "    fprintf(stderr, \"Failed to copy smh_coef_d: %s\\n\", hipGetErrorString(err));",
        "    return;", 
        "}",
        "err = hipMemcpyToSymbol(T_mid_d, T_mid_h, sizeof(double) * NSP_SK);",
        "if (err != hipSuccess) {",
        "    fprintf(stderr, \"Failed to copy T_mid_d: %s\\n\", hipGetErrorString(err));",
        "    return;",
        "}"
    ])
    
    function = f"""extern "C" {{
    void copyConstantsToDevice({(chr(10) + " "*27).join(params)})
    {{
        {(chr(10) + " "*8).join(copies)}
    }}

}}"""

    function += """
extern "C" {
    void allocate_intermediate_DeviceMemory(int ng) {
        hipError_t err;

        err = hipMalloc((void**)&rr_d, ng * NREACT_MECH * sizeof(double));

        err = hipMalloc((void**)&sigma_logC_r_d, ng * NREACT_MECH * sizeof(double));
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate device memory for sigma_logC_r_d: %s\\n", hipGetErrorString(err));
            hipFree(rr_d);
            return;
        }

        err = hipMalloc((void**)&sigma_logC_p_d, ng * NREACT_MECH * sizeof(double));
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate device memory for sigma_logC_p_d: %s\\n", hipGetErrorString(err));
            hipFree(rr_d);
            hipFree(sigma_logC_r_d);
            return;
        }

        err = hipMalloc((void**)&logEQK_d, ng * NREACT_MECH * sizeof(double));
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate device memory for logEQK_d: %s\\n", hipGetErrorString(err));
            hipFree(rr_d);
            hipFree(sigma_logC_r_d);
            hipFree(sigma_logC_p_d);
            return;
        }

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