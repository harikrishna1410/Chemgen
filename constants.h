#ifndef CONSTANTS_H
#define CONSTANTS_H
#include "hip/hip_runtime.h"
///****************mechanism constants************************
//number of reduced species
#define NSP_RED 9
//number of skeletal species
#define NSP_SK 9
//number of qssa species
#define NSP_QSSA 0
///these are total number of reactions in the mech
//these variables had to be divided by NREACT_PER_BLOCK
#define NREACT_MECH 23
#define NREACT_STD 16
#define NREACT_TROE 2
#define NREACT_THIRD 5
#define NREACT_PLOG 0
//
//maximum number of species involved in a reaction
//either on reactants side or products side
//MAX_SP should be divisible by NSPEC_PER_THREAD when using v3 parallelisation
#define MAX_SP 2
//twice of above
#define MAX_SP2 4
//max third body third body reactants
#define MAX_THIRD_BODIES 4
//when using more v2 and v3 parallelisation
//this had to divide all NREACT_* variables
#define NREACT_PER_BLOCK 16 //number of reactions solved per thread block
//this is used in v3 parallelisation
//MAX_SP should be divisible by NSP_PER_THREAD when using v3 parallelisation
//this is always some power of 2
#define NSP_PER_BLOCK 4
//max number of species per thread block
//ratio of MAX_SP/NSP_PER_THREAD should be divisible by 2
#define SP_PER_THREAD 1
//twice of above
#define SP2_PER_THREAD 2

const double RU = 83145100.0; //universal gas constant
const double SMALL = 1e-200; //a small value
const double PATM = 1013250.0; //atmospheric pressure
///****************mechanism constants************************

// Declare mechanism constants in constant memory
__device__ __constant__ double A_d[NREACT_MECH];
__device__ __constant__ double B_d[NREACT_MECH*2];
__device__ __constant__ int sk_map_d[MAX_SP*2*NREACT_MECH];
__device__ __constant__ double sk_coef_d[MAX_SP*2*NREACT_MECH];
__device__ __constant__ int map_r_d[MAX_SP*NREACT_MECH];
__device__ __constant__ double coef_r_d[MAX_SP*NREACT_MECH];
__device__ __constant__ int map_p_d[MAX_SP*NREACT_MECH];
__device__ __constant__ double coef_p_d[MAX_SP*NREACT_MECH];



static inline void copyConstantsToDevice(const double* A_h,
                           const double* B_h,
                           const int* sk_map_h,
                           const double* sk_coef_h,
                           const int* map_r_h,
                           const double* coef_r_h,
                           const int* map_p_h,
                           const double* coef_p_h)
{
    hipMemcpyToSymbol(A_d, A_h, sizeof(double) * NREACT_MECH);
    hipMemcpyToSymbol(B_d, B_h, sizeof(double) * NREACT_MECH * 2);
    hipMemcpyToSymbol(sk_map_d, sk_map_h, sizeof(int) * NREACT_MECH * MAX_SP * 2);
    hipMemcpyToSymbol(sk_coef_d, sk_coef_h, sizeof(double) * NREACT_MECH * MAX_SP * 2);
    hipMemcpyToSymbol(map_r_d, map_r_h, sizeof(int) * NREACT_MECH * MAX_SP);
    hipMemcpyToSymbol(coef_r_d, coef_r_h, sizeof(double) * NREACT_MECH * MAX_SP);
    hipMemcpyToSymbol(map_p_d, map_p_h, sizeof(int) * NREACT_MECH * MAX_SP);
    hipMemcpyToSymbol(coef_p_d, coef_p_h, sizeof(double) * NREACT_MECH * MAX_SP);
}


#endif
