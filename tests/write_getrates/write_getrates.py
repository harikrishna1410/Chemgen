import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',"..")))

from chemgen.parser import ckparser
from chemgen.chemistry import chemistry,chemistry_expressions
from chemgen.write_constants import *


abs_path = os.path.abspath(__file__)
# ck_file = os.path.join(os.path.dirname(abs_path), "../../ck_files/H2_burke/chem_without_CO_CO2_AR_HE.inp")
# therm_file = os.path.join(os.path.dirname(abs_path), "../../ck_files/H2_burke/therm.dat")
ck_file = os.path.join(os.path.dirname(abs_path), "../../ck_files/methane_NUI/chem.ske50_R1_R6_dup_fix.inp")
therm_file = os.path.join(os.path.dirname(abs_path), "../../ck_files/methane_NUI/therm.dat")

ckp = ckparser()
chem = chemistry(ck_file,ckp,therm_file=therm_file)

for vec,ext_vec in zip([True,False],["_i",""]):
    for lang,ext in zip(["fortran","python"],["f90","py"]):
        chem_expr =chemistry_expressions(chem,vec=vec,language=lang)
        filename = os.path.join(os.path.dirname(abs_path),f"getrates{ext_vec}.{ext}")
        chem_expr.write_expressions_to_file(filename)










