"""
This script writes all the constants to a file.
"""

from .constants import get_header_content, RU, SMALL, PATM
from .writer_utils import *
from .chemistry import chemistry
import sys
import numpy as np

space = " "
R_c = 1.987215575926745  # cal/mol/K

def get_arh_coef_lines(r_dict, troe=False):
    A_lines, B_lines, nr_calc = [], [], 0
    for rnum, reaction in r_dict.items():
        params = reaction["troe"]["low"] if troe else reaction["arh"]
        A, beta, Ea = params

        aline = f"{space}{np.log(A):21.15E},& \n".replace("E", "D")
        bline = f"{space}{'+' if beta > 0 else '-'}{abs(beta):21.15E},".replace("E", "D")
        bline += f"{'-' if Ea > 0 else '+'}{abs(Ea)/R_c:21.15E},&\n".replace("E", "D")

        A_lines.append(aline)
        B_lines.append(bline)
        nr_calc += 1
    if len(A_lines) > 0:
        A_lines[-1] = A_lines[-1].replace(",", "")
        B_lines[-1] = B_lines[-1][:-3] + "&\n"
    return nr_calc, A_lines, B_lines

def write_arrhenius_constants(chem: chemistry):
    new_lines = []
    A_lines, B_lines, A0_troe_lines, B0_troe_lines = [], [], [], []
    nr_calc, nr_troe = 0, 0

    for reaction_type in ['standard', 'troe', 'third_body', 'plog']:
        nr, A, B = get_arh_coef_lines(chem.get_reactions_by_type(reaction_type))
        A_lines.extend(A)
        B_lines.extend(B)
        nr_calc += nr

        if reaction_type == 'troe':
            nr_troe, A0, B0 = get_arh_coef_lines(chem.get_reactions_by_type('troe'), troe=True)
            A0_troe_lines.extend(A0)
            B0_troe_lines.extend(B0)

    new_lines = add_new_array("real", "A_h", nr_calc, A_lines, new_lines)
    new_lines = add_new_array("real", "B_h", 2*nr_calc, B_lines, new_lines)
    new_lines = add_new_array("real", "A0_troe_h", nr_troe, A0_troe_lines, new_lines)
    new_lines = add_new_array("real", "B0_troe_h", 2*nr_troe, B0_troe_lines, new_lines)

    return new_lines

def write_maps(chem: chemistry):
    new_lines = []
    r_dict = chem.reactions
    max_sp = chem.find_max_specs()[0] // 2

    sk_map, map_r, map_p = [], [], []
    
    for reaction in r_dict.values():
        for species_list, map_list in [(reaction['reacts'], map_r), (reaction['prods'], map_p)]:
            sk_map.extend([chem.species_index(s) for s in species_list])
            map_list.extend([chem.reduced_species_index(s) for s in species_list])
            pad_length = max_sp - len(species_list)
            sk_map.extend([0] * pad_length)
            map_list.extend([0] * pad_length)

    for map_name, map_data in [("sk_map_h", sk_map), ("map_r_h", map_r), ("map_p_h", map_p)]:
        lines = []
        curr_line = ""
        for m in map_data:
            curr_line, lines = append_new_str(f"{m},", curr_line, lines)
        new_lines = add_new_array("integer", map_name, len(map_data), lines, new_lines)
    return new_lines

def write_coefficients(chem: chemistry):
    new_lines = []
    r_dict = chem.reactions
    max_sp = chem.find_max_specs()[0] // 2

    sk_coef, coef_r, coef_p = [], [], []
    
    for reaction in r_dict.values():
        for coef_list, species_dict, sign in [(sk_coef, reaction['reacts'], -1), (sk_coef, reaction['prods'], 1),
                                              (coef_r, reaction['reacts'], 1), (coef_p, reaction['prods'], 1)]:
            coef_list.extend([species_dict[s] * sign for s in species_dict])
            coef_list.extend([0.0] * (max_sp - len(species_dict)))

    for coef_name, coef_array in [("sk_coef_h", sk_coef), ("coef_r_h", coef_r), ("coef_p_h", coef_p)]:
        lines = []
        curr_line = ""
        for c in coef_array:
            curr_line, lines = append_new_str(f"{c:.1f},", curr_line, lines)
        new_lines = add_new_array("real", coef_name, len(coef_array), lines, new_lines)
    
    return new_lines

def write_constants_header(dirname, chem: chemistry):
    header_content = get_header_content(chem)
    
    with open(f"{dirname}/constants.h", "w") as f:
        f.write(header_content)
    
    return

def write_coef_module(dirname, chem: chemistry):
    with open(f"{dirname}/coef_m.f90", "w") as f:
        f.write("module coef_m\n")
        f.write("  implicit none\n\n")
        
        # Write Arrhenius constants
        arrhenius_lines = write_arrhenius_constants(chem)
        f.writelines(arrhenius_lines)
        
        # Write maps
        map_lines = write_maps(chem)
        f.writelines(map_lines)
        
        # Write coefficients
        coef_lines = write_coefficients(chem)
        f.writelines(coef_lines)
        
        f.write("end module coef_m\n")
    
    return