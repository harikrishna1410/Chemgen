import numpy as np
import cantera as ct
from getrates import getrates

# Set up initial conditions
phi = 1.0
T = 1200  # K
P = ct.one_atm  # 1 atm

# Create gas object
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
wdot = np.zeros_like(Y)
P_cgs = P * 10  # Convert from Pa to dyne/cm^2
kf, kb, rr = getrates(T, Y, P_cgs, wdot)
custom_wdot.append(wdot)

# cantera_wdot.append(gas.net_production_rates)
# # Run simulation
# for t in times:
#     for reaction_index in zero_reactions:
#         gas.set_multiplier(0.0, reaction_index - 1)  # Cantera uses 0-indexed reactions
#     sim.advance(t)
#     # Get state
#     Y = r.thermo.Y
#     T = r.T
#     P = r.thermo.P
    
#     # Get Cantera species production rates
#     cantera_wdot.append(gas.net_production_rates)
    
#     # Get custom species production rates
#     wdot = np.zeros_like(Y)
#     P_cgs = P * 10  # Convert from Pa to dyne/cm^2
#     kf, kb, rr = getrates(T, Y, P_cgs, wdot)
#     custom_wdot.append(wdot)

# Convert to numpy arrays
cantera_wdot = np.array(cantera_wdot)
custom_wdot = np.array(custom_wdot)

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
        
    # Print detailed comparison for the last time step
    print("\nDetailed comparison for the last time step:")
    species_names = gas.species_names
    for i, (cw, custw) in enumerate(zip(cantera_wdot[-1], custom_wdot[-1])):
        print(f"Species {species_names[i]}:")
        print(f"  Cantera wdot: {cw:.4e}")
        print(f"  Custom wdot:  {custw:.4e}")
        print(f"  Relative diff: {rel_diff[-1, i]:.4e}")
        print()

# Run comparison
compare_wdot(cantera_wdot, custom_wdot)
