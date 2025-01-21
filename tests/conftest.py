# conftest.py

def pytest_addoption(parser):
    parser.addoption(
        "--temperature",
        action="store",
        default=1200,
        type=float,
        help="temperature in K"
    )
    parser.addoption(
        "--pressure",
        action="store",
        default=101325,  # one atmosphere in Pa
        type=float,
        help="pressure in Pa"
    )
    parser.addoption(
        "--phi",
        action="store",
        default=0.5,  # one atmosphere in Pa
        type=float,
        help="equivalence ratio"
    )
    parser.addoption(
        "--fuel",
        action="store",
        default="H2:1",
        help="fuel string in format 'species:mole_fraction'"
    )
    parser.addoption(
        "--oxidiser",
        action="store",
        default="O2:1,N2:3.76",
        help="oxidiser string in format 'species:mole_fraction,species:mole_fraction'"
    )
    parser.addoption(
        "--mech_file_ck",
        action="store",
        default="ck_files/H2_burke/chem_without_CO_CO2_AR_HE.inp",
        help="path to chemkin mechanism file"
    )
    parser.addoption(
        "--therm_file",
        action="store",
        default="ck_files/H2_burke/therm.dat",
        help="path to thermodynamic data file"
    )
    parser.addoption(
        "--language",
        action="store",
        default="python",
        help="desired language for generated code"
    )