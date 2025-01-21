import numpy as np
import cantera as ct
import pytest
import tempfile
import os
import sys
from chemgen.parser import ckparser
from chemgen.chemistry import chemistry, chemistry_expressions

@pytest.fixture
def set_mech_and_state(request):
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        mech_file_ck = request.config.getoption("--mech_file_ck")
        therm_file = request.config.getoption("--therm_file")
        language = request.config.getoption("--language")
        
        # Convert chemkin file to cantera yaml format
        mech_file_ct = os.path.splitext(mech_file_ck)[0] + '.yaml'
        
        # Get temperature and pressure from options
        T = request.config.getoption("--temperature")
        P = request.config.getoption("--pressure")
        
        # Parse fuel and oxidiser strings
        fuel = request.config.getoption("--fuel")
        oxidiser = request.config.getoption("--oxidiser")
        reactants = {"fu": fuel, "ox": oxidiser}
        phi = 0.5

        ckp = ckparser()
        chem = chemistry(mech_file_ck, ckp, therm_file=therm_file)
    
        # Create chemistry expressions object with specified options
        chem_expr = chemistry_expressions(chem,
                                       vec=False,
                                       omp=False,
                                       mod=False,
                                       language=language)
    
        # Write expressions to file
        source_file = os.path.join(tmpdir, f'getrates_{language}')
        ext={'python': 'py', 'fortran': 'f90'}
        chem_expr.write_expressions_to_file(f'{source_file}.{ext[language]}',
                                      write_rtypes_together=False,
                                      input_MW=False)
        
        sys.path.insert(0, tmpdir)
        if language == 'fortran':
            import subprocess
            import numpy.f2py
            # Compile Fortran code using f2py
            subprocess.check_call(['f2py', '-c', f'{source_file}.f90', '-m', 'getrates_fortran'])
            from getrates_fortran import getrates
        else:
            from getrates_python import getrates
        
        yield getrates,mech_file_ct,reactants,T,P,phi,language
        
        # Cleanup
        sys.path.remove(tmpdir)

def test_getrates(set_mech_and_state):
    getrates,mech_file_ct,reactants,T,P,phi,lang = set_mech_and_state

    # Create gas object
    gas = ct.Solution(mech_file_ct)

    # Set the gas state
    gas.set_equivalence_ratio(phi, reactants["fu"], reactants["ox"])
    Y = gas.Y
    gas.TP = T, P

    # Get Cantera production rates
    cantera_wdot = gas.net_production_rates
    P_cgs = P * 10  # Convert from Pa to dyne/cm^2

    # Test scalar version
    wdot = np.zeros_like(np.zeros(gas.n_species))
    if lang == 'fortran':
        wdot = getrates(P_cgs, T, Y)
    else:
        getrates(T, Y, P_cgs, wdot)

    # Compare results
    tolerance = 1e-6
    
    # Test scalar version
    abs_diff = np.abs(cantera_wdot*1e-3 - wdot)
    rel_diff = abs_diff / (np.abs(cantera_wdot) + 1e-20)
    max_rel_diff = np.max(rel_diff)
    assert max_rel_diff < tolerance, f"Scalar version failed: max relative difference {max_rel_diff}"
