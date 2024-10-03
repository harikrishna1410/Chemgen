import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chemgen.parser import ckparser
from chemgen.chemistry import chemistry
from chemgen.write_constants import *

abs_path = os.path.abspath(__file__)
ck_file = os.path.join(os.path.dirname(abs_path), "../ck_files/H2_burke/chem_without_CO_CO2_AR_HE.inp")
therm_file = os.path.join(os.path.dirname(abs_path), "../ck_files/H2_burke/therm.dat")

ckp = ckparser()
chem = chemistry(ck_file,therm_file,ckp)

#print(chem.get_reactions_by_type("troe"))
write_constants_header("./", chem)
write_coef_module("./", chem)

#reacts = chem.reactions

#for rnum in reacts.keys():
#    print(rnum,reacts[rnum]["eqn"])
#    print(reacts[rnum]["reacts"])
#    print(reacts[rnum]["prods"])












