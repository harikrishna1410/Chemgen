
#ifdef HIP_CHEM_V3
#include "constants_v3.h"
#elif HIP_CHEM_V2
#include "constants_v2.h"
#elif HIP_CHEM_V1
#include "constants_v1.h"
#endif

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
