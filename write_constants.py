"""
this script writes all the constants to a file.
NSP_RED: number of species in the reduced mechanism
NSP_SK: number of species in the skeletal mechanism
NSP_QSSA: number of species in QSSA approximation
NREACT_MECH: number of reactions in the mechanism
NREACT_STD: number of standard arhhennius reactions
NREACT_TROE: number of troe reactions
NREACT_THIRD: number of third body reactions
NREACT_PLOG: number of plog reactions
//MAX_SP should be divisible by NSPEC_PER_THREAD when using v3 parallelisation
MAX_SP:maximum number of species involved in a reaction. either on reactants side or products side
//twice of above
MAX_SP2: twice of above
//max third body third body reactants
MAX_THIRD_BODIES: max third body third body reactants
//when using more v2 and v3 parallelisation
//this had to divide all NREACT_* variables
NREACT_PER_BLOCK: number of reactions solved per thread block
//this is used in v3 parallelisation
//MAX_SP should be divisible by NSP_PER_THREAD when using v3 parallelisation
//this is always some power of 2
NSP_PER_BLOCK: max number of species per thread block
SP_PER_THREAD: ratio of MAX_SP/NSP_PER_BLOCK
//twice of above
SP2_PER_THREAD: 

A: Arrhennius constants
B[2]: beta and Ea/R
Sk_map: skeletal mechanism map
SK_coef: skeletal mechanism coefficients
map_r: reactants map of the reduced mechanism
map_p: products map of the reduced mechanism
coef_r: reactants coefficients of the reduced mechanism
coef_p: products coefficients of the reduced mechanism
"""

