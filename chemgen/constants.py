from .chemistry import chemistry
from pint import UnitRegistry

# Universal constants
RU = 8.31451e7  # universal gas constant
SMALL = 1.0e-200  # a small value
PATM = 1.01325e6  # atmospheric pressure

# Default parallelization constants
DEFAULT_NREACT_PER_BLOCK = 1
DEFAULT_NSP_PER_BLOCK = 1

# Element weights
ELEM_WT = {
    "H": 1.007969975471497E0,
    "O": 1.599940013885498E1,
    "C": 1.201115036010742E1,
    "N": 1.400669956207276E1
}

# Constant memory declarations
CONSTANT_MEMORY_DECLARATIONS = """
// Declare mechanism constants in constant memory
__device__ __constant__ double A_d[NREACT_MECH];
__device__ __constant__ double B_d[NREACT_MECH*2];
__device__ __constant__ int sk_map_d[MAX_SP*2*NREACT_MECH];
__device__ __constant__ double sk_coef_d[MAX_SP*2*NREACT_MECH];
__device__ __constant__ int map_r_d[MAX_SP*NREACT_MECH];
__device__ __constant__ double coef_r_d[MAX_SP*NREACT_MECH];
__device__ __constant__ int map_p_d[MAX_SP*NREACT_MECH];
__device__ __constant__ double coef_p_d[MAX_SP*NREACT_MECH];
__device__ __constant__ double mw_d[NSP_RED];
__device__ __constant__ double smh_coef_d[NSP_SK*14];
__device__ __constant__ double T_mid_d[NSP_SK];
"""

# Copy constants to device function
COPY_CONSTANTS_TO_DEVICE_FUNC = """
extern "C" {
    void copyConstantsToDevice(const double* A_h,
                           const double* B_h,
                           const int* sk_map_h,
                           const double* sk_coef_h,
                           const int* map_r_h,
                           const double* coef_r_h,
                           const int* map_p_h,
                           const double* coef_p_h,
                           const double* mw_h,
                           const double* smh_coef_h,
                           const double* T_mid_h)
    {
        hipMemcpyToSymbol(A_d, A_h, sizeof(double) * NREACT_MECH);
        hipMemcpyToSymbol(B_d, B_h, sizeof(double) * NREACT_MECH * 2);
        hipMemcpyToSymbol(sk_map_d, sk_map_h, sizeof(int) * NREACT_MECH * MAX_SP * 2);
        hipMemcpyToSymbol(sk_coef_d, sk_coef_h, sizeof(double) * NREACT_MECH * MAX_SP * 2);
        hipMemcpyToSymbol(map_r_d, map_r_h, sizeof(int) * NREACT_MECH * MAX_SP);
        hipMemcpyToSymbol(coef_r_d, coef_r_h, sizeof(double) * NREACT_MECH * MAX_SP);
        hipMemcpyToSymbol(map_p_d, map_p_h, sizeof(int) * NREACT_MECH * MAX_SP);
        hipMemcpyToSymbol(coef_p_d, coef_p_h, sizeof(double) * NREACT_MECH * MAX_SP);
        hipMemcpyToSymbol(mw_d, mw_h, sizeof(double) * NSP_RED);
        hipMemcpyToSymbol(smh_coef_d, smh_coef_h, sizeof(double) * NSP_SK * 14);
        hipMemcpyToSymbol(T_mid_d, T_mid_h, sizeof(double) * NSP_SK);
    }
}
"""

# Constant memory declarations
CONSTANT_MEMORY_DECLARATIONS_ROCBLAS = """
// Declare mechanism constants in global memory
__device__ double* A_d;
__device__ double* B_d;
__device__ double* sk_coef_d;
__device__ double* coef_r_d;
__device__ double* coef_p_d;
__device__ double* wdot_coef_d;
__device__ __constant__ double mw_d[NSP_RED];
__device__ __constant__ double smh_coef_d[NSP_SK*14];
__device__ __constant__ double T_mid_d[NSP_SK];

//intermediate arrays
__device__ double *rr_d;
__device__ double *sigma_logC_r_d; 
__device__ double *sigma_logC_p_d;
__device__ double *logEQK_d;
"""

# Copy constants to device function
COPY_CONSTANTS_TO_DEVICE_FUNC_ROCBLAS = """
extern "C" {
    hipError_t allocate_intermediate_DeviceMemory(int ng) {
        hipError_t err;

        err = hipMalloc((void**)&rr_d, ng * NREACT_MECH * sizeof(double));
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate device memory for rr_d: %s\\n", hipGetErrorString(err));
            return err;
        }

        err = hipMalloc((void**)&sigma_logC_r_d, ng * NREACT_MECH * sizeof(double));
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate device memory for sigma_logC_r_d: %s\\n", hipGetErrorString(err));
            hipFree(rr_d);
            return err;
        }

        err = hipMalloc((void**)&sigma_logC_p_d, ng * NREACT_MECH * sizeof(double));
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate device memory for sigma_logC_p_d: %s\\n", hipGetErrorString(err));
            hipFree(rr_d);
            hipFree(sigma_logC_r_d);
            return err;
        }

        err = hipMalloc((void**)&logEQK_d, ng * NREACT_MECH * sizeof(double));
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate device memory for logEQK_d: %s\\n", hipGetErrorString(err));
            hipFree(rr_d);
            hipFree(sigma_logC_r_d);
            hipFree(sigma_logC_p_d);
            return err;
        }

        return hipSuccess;
    }
}

extern "C" {
    void copyConstantsToDevice(const double* A_h,
                           const double* B_h,
                           const double* sk_coef_h,
                           const double* coef_r_h,
                           const double* coef_p_h,
                           const double* wdot_coef_h,
                           const double* mw_h,
                           const double* smh_coef_h,
                           const double* T_mid_h)
    {
        hipError_t err;

        err = hipMalloc((void**)&A_d, sizeof(double) * NREACT_MECH);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate A_d: %s\\n", hipGetErrorString(err));
            return;
        }

        err = hipMalloc((void**)&B_d, sizeof(double) * NREACT_MECH * 2);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate B_d: %s\\n", hipGetErrorString(err));
            hipFree(A_d);
            return;
        }

        err = hipMalloc((void**)&sk_coef_d, sizeof(double) * NREACT_MECH * NSP_SK);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate sk_coef_d: %s\\n", hipGetErrorString(err));
            hipFree(A_d);
            hipFree(B_d);
            return;
        }

        err = hipMalloc((void**)&coef_r_d, sizeof(double) * NREACT_MECH * NSP_RED);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate coef_r_d: %s\\n", hipGetErrorString(err));
            hipFree(A_d);
            hipFree(B_d);
            hipFree(sk_coef_d);
            return;
        }

        err = hipMalloc((void**)&coef_p_d, sizeof(double) * NREACT_MECH * NSP_RED);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate coef_p_d: %s\\n", hipGetErrorString(err));
            hipFree(A_d);
            hipFree(B_d);
            hipFree(sk_coef_d);
            hipFree(coef_r_d);
            return;
        }

        err = hipMalloc((void**)&wdot_coef_d, sizeof(double) * NREACT_MECH * NSP_RED);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate wdot_coef_d: %s\\n", hipGetErrorString(err));
            hipFree(A_d);
            hipFree(B_d);
            hipFree(sk_coef_d);
            hipFree(coef_r_d);
            hipFree(coef_p_d);
            return;
        }

        err = hipMemcpy(A_d, A_h, sizeof(double) * NREACT_MECH, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to copy A_d: %s\\n", hipGetErrorString(err));
            goto cleanup;
        }

        err = hipMemcpy(B_d, B_h, sizeof(double) * NREACT_MECH * 2, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to copy B_d: %s\\n", hipGetErrorString(err));
            goto cleanup;
        }

        err = hipMemcpy(sk_coef_d, sk_coef_h, sizeof(double) * NREACT_MECH * NSP_SK, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to copy sk_coef_d: %s\\n", hipGetErrorString(err));
            goto cleanup;
        }

        err = hipMemcpy(coef_r_d, coef_r_h, sizeof(double) * NREACT_MECH * NSP_RED, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to copy coef_r_d: %s\\n", hipGetErrorString(err));
            goto cleanup;
        }

        err = hipMemcpy(coef_p_d, coef_p_h, sizeof(double) * NREACT_MECH * NSP_RED, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to copy coef_p_d: %s\\n", hipGetErrorString(err));
            goto cleanup;
        }

        err = hipMemcpy(wdot_coef_d, wdot_coef_h, sizeof(double) * NREACT_MECH * NSP_RED, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to copy wdot_coef_d: %s\\n", hipGetErrorString(err));
            goto cleanup;
        }
        hipMemcpyToSymbol(mw_d, mw_h, sizeof(double) * NSP_RED);
        hipMemcpyToSymbol(smh_coef_d, smh_coef_h, sizeof(double) * NSP_SK * 14);
        hipMemcpyToSymbol(T_mid_d, T_mid_h, sizeof(double) * NSP_SK);

        return;

    cleanup:
        hipFree(A_d);
        hipFree(B_d);
        hipFree(sk_coef_d);
        hipFree(coef_r_d);
        hipFree(coef_p_d);
        hipFree(wdot_coef_d);
    }
}
"""

def get_header_content(chem: chemistry,
                       parallel_level=1,
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
{CONSTANT_MEMORY_DECLARATIONS if not rocblas else CONSTANT_MEMORY_DECLARATIONS_ROCBLAS}

{COPY_CONSTANTS_TO_DEVICE_FUNC if not rocblas else COPY_CONSTANTS_TO_DEVICE_FUNC_ROCBLAS}
#endif
"""
    return header_content