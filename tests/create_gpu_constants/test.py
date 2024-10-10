import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chemgen.parser import ckparser
from chemgen.chemistry import chemistry
from chemgen.write_constants import *


abs_path = os.path.abspath(__file__)
ck_file = os.path.join(os.path.dirname(abs_path), "../ck_files/methane_NUI/chem.ske50_R1_R6_dup_fix.inp")
therm_file = os.path.join(os.path.dirname(abs_path), "../ck_files/methane_NUI/therm.dat")

ckp = ckparser()
chem = chemistry(ck_file,ckp,therm_file=therm_file)

# troe_reactions = chem.get_reactions_by_type("troe")
# for i, r in troe_reactions.items():
#     print(f"Reaction {i}: {r['eqn']}")

# print("\n\n")
# troe_reactions = chem.get_reactions_by_type("third_body")
# for i, r in troe_reactions.items():
#     print(f"Reaction {i}: {r['eqn']}")


# #print(chem.get_reactions_by_type("troe"))
write_coef_module(os.path.dirname(abs_path), chem,parallel_level=1)
write_constants_header(os.path.dirname(abs_path), chem,parallel_level=1,veclen=16)


write_coef_module(os.path.dirname(abs_path), chem,parallel_level=2,nreact_per_block=4)
write_constants_header(os.path.dirname(abs_path), chem,parallel_level=2,veclen=16,nreact_per_block=4)


write_coef_module(os.path.dirname(abs_path), chem,parallel_level=3,nreact_per_block=4)
write_constants_header(os.path.dirname(abs_path), chem,parallel_level=3,veclen=16,nreact_per_block=4)


#reacts = chem.reactions

#for rnum in reacts.keys():
#    print(rnum,reacts[rnum]["eqn"])
#    print(reacts[rnum]["reacts"])
#    print(reacts[rnum]["prods"])












