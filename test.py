from utils import *

ckp = ckparser("./chem.ske50.inp","./thermo.dat")

reacts = ckp.parse_reactions()

for rnum in reacts.keys():
    print(reacts[rnum]["eqn"])
    print(reacts[rnum]["reacts"])
    print(reacts[rnum]["prods"])

print(len(reacts.keys()))
