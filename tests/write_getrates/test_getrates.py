import numpy as np
import cantera as ct
from getrates import getrates as getrates_python
from getrates_i import getrates as getrates_python_vec
from getrates_ftn_module import getrates as getrates_ftn
from getrates_ftn_module_vec import getrates as getrates_ftn_vec

# Set up initial conditions
phi = 1.0
T = 1200  # K
P = ct.one_atm  # 1 atm

# Create gas object
# gas = ct.Solution('CH4_NUI_sk50.yaml')
gas = ct.Solution('H2_burke.yaml')

# Set the gas state
gas.set_equivalence_ratio(phi, 'H2', 'O2:1.0, N2:3.76')
gas.Y = np.array([1.0/gas.n_species]*gas.n_species)
gas.TP = T, P

# # Set some reactions to zero in Cantera
# zero_reactions = range(2,23)  # Reactions to be set to zero (1-indexed)
# # Modify the gas object to set specified reactions to zero
# for reaction_index in zero_reactions:
#     gas.set_multiplier(0.0, reaction_index - 1)  # Cantera uses 0-indexed reactions

# Create reactor network
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

# Time array for integration
t_end = 1e-3
dt = 1e-5
times = np.arange(0, t_end, dt)

# Lists to store results
cantera_wdot = []
custom_wdot = []

cantera_wdot.append(gas.net_production_rates)
Y = gas.Y
print(Y)

P_cgs = P * 10  # Convert from Pa to dyne/cm^2
veclen = 10
wdot_ftn = np.zeros_like(Y)
wdot_python = np.zeros_like(Y)
ickwrk = np.zeros((10,))
rckwrk = np.zeros((10,))

# wdot_ftn = getrates_ftn(1, T, Y, P_cgs)
veclen = 1
wdot_python_vec = np.zeros((veclen,gas.n_species))
wdot_ftn_vec = np.zeros((veclen,gas.n_species))
T_array = np.zeros((veclen,))
T_array[:] = T
P_array = np.zeros((veclen,))
P_array[:] = P_cgs
Y_array = np.zeros_like(wdot_python_vec)
for i in range(veclen):
    Y_array[i,:] = Y[:]
getrates_python(T, Y, P_cgs , wdot_python)
getrates_python_vec(veclen,T_array, Y_array, P_array , wdot_python_vec)
wdot_ftn = getrates_ftn(P_cgs, T, Y , ickwrk, rckwrk)
wdot_ftn_vec = getrates_ftn_vec(T_array, Y_array, P_array,veclen)

custom_wdot.append(wdot_python)
custom_wdot.append(wdot_python_vec[0])
custom_wdot.append(wdot_ftn)
custom_wdot.append(wdot_ftn_vec[0])


# Compare species production rates
def compare_wdot(cantera_wdot, custom_wdot, tolerance=1e-6):
    abs_diff = np.abs(cantera_wdot*1e-3 - custom_wdot)
    rel_diff = abs_diff / (np.abs(cantera_wdot) + 1e-20)
    max_rel_diff = np.max(rel_diff)
    
    print(f"Maximum relative difference: {max_rel_diff:.2e}")
    
    if max_rel_diff < tolerance:
        print("Test passed: Custom wdot matches Cantera wdot within tolerance.")
    else:
        print("Test failed: Custom wdot differs from Cantera wdot beyond tolerance.")
        
    # # Print detailed comparison for the last time step
    # print("\nDetailed comparison for the last time step:")
    # species_names = gas.species_names
    # for i, (cw, custw) in enumerate(zip(cantera_wdot[0], custom_wdot)):
    #     print(f"Species {species_names[i]}:")
    #     print(f"  Cantera wdot: {cw:.4e}")
    #     print(f"  Custom wdot:  {custw:.4e}")
    #     print(f"  Relative diff: {rel_diff[-1, i]:.4e}")
    #     print()

# Run comparison
for i in range(len(custom_wdot)):
    print(f"index {i}")
    compare_wdot(cantera_wdot[0], custom_wdot[i])
