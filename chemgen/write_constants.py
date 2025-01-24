"""
This script writes all the constants to a file.
"""

from .constants import get_header_content, RU, SMALL, PATM
from .writer_utils import *
from .chemistry import chemistry
import sys
import numpy as np

space = " "
R_c = 1.9872155832  # cal/mol/K

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

def write_arrhenius_constants(chem: chemistry,parallel_level=1,nreact_per_block=None):
    if parallel_level > 1:
        assert nreact_per_block is not None
    
    new_lines = []

    nr_calc, nr_troe = 0, 0

    for reaction_type in ['standard', 'troe', 'third_body', 'plog']:
        A_lines, B_lines, A0_troe_lines, B0_troe_lines = [], [], [], []
        reactions = chem.get_reactions_by_type(reaction_type)
        if(len(reactions.keys())==0):
            continue
        if nreact_per_block is not None:
            if len(reactions)%nreact_per_block != 0:
                for i in range(0,nreact_per_block - len(reactions)%nreact_per_block):
                    print("padding mech",i)
                    chem.add_dummy_reaction(reaction_type)
        reactions = chem.get_reactions_by_type(reaction_type)
        nr, A, B = get_arh_coef_lines(reactions)
        A_lines.extend(A)
        B_lines.extend(B)
        nr_calc += nr

        if reaction_type == 'troe':
            nr_troe, A0, B0 = get_arh_coef_lines(reactions, troe=True)
            A0_troe_lines.extend(A0)
            B0_troe_lines.extend(B0)
        
        type_suffix = "" if reaction_type == "standard" else \
                     "_inf_troe" if reaction_type == "troe" else \
                     "_third" if reaction_type == "third_body" else "_plog"

        new_lines = add_new_array("real", f"A{type_suffix}_h", nr, A_lines, new_lines)
        new_lines = add_new_array("real", f"B{type_suffix}_h", 2*nr, B_lines, new_lines)

        if(reaction_type == "troe"):
            new_lines = add_new_array("real", f"A_0_troe_h", nr_troe, A0_troe_lines, new_lines)
            new_lines = add_new_array("real", "B_0_troe_h", 2*nr_troe, B0_troe_lines, new_lines)

    return new_lines


def write_molwts(chem: chemistry,parallel_level=1,nreact_per_block=None):
    if parallel_level > 1:
        assert nreact_per_block is not None
    
    new_lines = []
    mw_lines = []

    for sp in chem.species_dict.values():
        line = f"{space}{sp.molecular_weight:21.15E},& \n".replace("E", "D")
        mw_lines.append(line)
    mw_lines[-1] = mw_lines[-1][:-4]+"&\n"
    new_lines = add_new_array("real", "mw_h", chem.n_species_red, mw_lines, new_lines)
    return new_lines

def write_thermo_data(chem: chemistry,parallel_level=1,nreact_per_block=None):
    if parallel_level > 1:
        assert nreact_per_block is not None
    
    new_lines = []
    thermo_lines = []
    T_mids = []

    for sp in chem.species_dict.values():
        low = sp.input_data["thermo"]["data"][0]
        high = sp.input_data["thermo"]["data"][1]
        T_mid = sp.input_data["thermo"]["temperature-ranges"][1]

        T_mids.append(f"{T_mid:.2E},& \n".replace("E","D"))

        # Low temperature coefficients
        #0
        line = f"{space}{low[6]-low[0]:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #1
        line = f"{space}{low[0]:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #2
        line = f"{space}{low[1]/2.0:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #3
        line = f"{space}{low[2]/6.0:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #4
        line = f"{space}{low[3]/12.0:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #5
        line = f"{space}{low[4]/20.0:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #6
        line = f"{space}{-low[5]:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)

        # High temperature coefficients
        #0
        line = f"{space}{high[6]-high[0]:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #1
        line = f"{space}{high[0]:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #2
        line = f"{space}{high[1]/2.0:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #3
        line = f"{space}{high[2]/6.0:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #4
        line = f"{space}{high[3]/12.0:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #5
        line = f"{space}{high[4]/20.0:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
        #6
        line = f"{space}{-high[5]:21.15E},& \n".replace("E", "D")
        thermo_lines.append(line)
    thermo_lines[-1] = thermo_lines[-1][:-4]+"&\n"
    T_mids[-1] = T_mids[-1][:-4]+"&\n"
    new_lines = add_new_array("real", "T_mid_h", chem.n_species_sk, T_mids, new_lines)
    new_lines = add_new_array("real", "smh_coef_h", chem.n_species_sk*14, thermo_lines, new_lines)
    return new_lines

def write_maps(chem: chemistry, parallel_level=1, nreact_per_block=None):
    if parallel_level > 1:
        assert nreact_per_block is not None
    
    new_lines = []
    r_dict = chem.reactions
    max_sp = chem.find_max_specs(parallel_level>2)

    for reaction_type in ['standard', 'troe', 'third_body', 'plog']:
        sk_map, map_r, map_p = [], [], []
        reactions = chem.get_reactions_by_type(reaction_type)
        if(len(reactions.keys())==0):
            continue
        if nreact_per_block is not None:    
            if len(reactions)%nreact_per_block != 0:
                for i in range(0,nreact_per_block - len(reactions)%nreact_per_block):
                    chem.add_dummy_reaction(reaction_type)
        reactions = chem.get_reactions_by_type(reaction_type)
        for reaction in reactions.values():
            for species_list, map_list in [(reaction['reacts'], map_r), (reaction['prods'], map_p)]:
                sk_map.extend([chem.species_index(s) for s in species_list])
                map_list.extend([chem.reduced_species_index(s) for s in species_list])
                pad_length = max_sp - len(species_list)
                sk_map.extend([0] * pad_length)
                map_list.extend([0] * pad_length)

        type_suffix = "" if reaction_type == "standard" else \
                     "_troe" if reaction_type == "troe" else \
                     "_third" if reaction_type == "third_body" else "_plog"
                     
        for map_name, map_data in [(f"sk_map{type_suffix}_h", sk_map), 
                                  (f"map_r{type_suffix}_h", map_r), 
                                  (f"map_p{type_suffix}_h", map_p)]:
            lines = []
            curr_line = ""
            for m in map_data:
                curr_line, lines = append_new_str(f"{m},", curr_line, lines)
            if(len(curr_line) > 0):
                lines.append(curr_line[:-1]+"&\n")
            new_lines = add_new_array("integer", map_name, len(map_data), lines, new_lines)
    return new_lines

def write_coefficients(chem: chemistry, parallel_level=1, nreact_per_block=None):
    if parallel_level > 1:
        assert nreact_per_block is not None
    
    new_lines = []
    r_dict = chem.reactions
    max_sp = chem.find_max_specs(parallel_level>2)

    for reaction_type in ['standard', 'troe', 'third_body', 'plog']:
        sk_coef, coef_r, coef_p = [], [], []
        reactions = chem.get_reactions_by_type(reaction_type)
        if(len(reactions.keys())==0):
            continue
        if nreact_per_block is not None:    
            if len(reactions)%nreact_per_block != 0:
                for i in range(0,nreact_per_block - len(reactions)%nreact_per_block):
                    chem.add_dummy_reaction(reaction_type)
        reactions = chem.get_reactions_by_type(reaction_type)
        for reaction in reactions.values():
            for coef_list, species_dict, sign in [(sk_coef, reaction['reacts'], -1), (sk_coef, reaction['prods'], 1),
                                                  (coef_r, reaction['reacts'], 1), (coef_p, reaction['prods'], 1)]:
                coef_list.extend([species_dict[s] * sign for s in species_dict])
                coef_list.extend([0.0] * (max_sp - len(species_dict)))

        type_suffix = "" if reaction_type == "standard" else \
                     "_troe" if reaction_type == "troe" else \
                     "_third" if reaction_type == "third_body" else "_plog"
                     
        for coef_name, coef_array in [(f"sk_coef{type_suffix}_h", sk_coef), 
                                     (f"coef_r{type_suffix}_h", coef_r), 
                                     (f"coef_p{type_suffix}_h", coef_p)]:
            lines = []
            curr_line = ""
            for c in coef_array:
                curr_line, lines = append_new_str(f"{c:.1f},", curr_line, lines)
            if(len(curr_line) > 0):
                lines.append(curr_line[:-1]+"&\n")
            new_lines = add_new_array("real", coef_name, len(coef_array), lines, new_lines)
    
    return new_lines

def write_third_body_efficiencies(chem: chemistry, parallel_level=1, nreact_per_block=None):
    if parallel_level > 1:
        assert nreact_per_block is not None
    
    new_lines = []
    r_dict = chem.reactions
    max_sp = chem.find_max_specs(parallel_level>2)

    for reaction_type in ['troe', 'third_body']:
        coef_array = []
        reactions = chem.get_reactions_by_type(reaction_type)
        if(len(reactions.keys())==0):
            continue
        if nreact_per_block is not None:    
            if len(reactions)%nreact_per_block != 0:
                for i in range(0,nreact_per_block - len(reactions)%nreact_per_block):
                    chem.add_dummy_reaction(reaction_type)
        reactions = chem.get_reactions_by_type(reaction_type)
        for reaction in reactions.values():
            for spec in chem.reduced_species:
                coef_array.append(1.0+reaction.get("third_body",{}).get(spec,0.0))

        type_suffix = "" if reaction_type == "standard" else \
                     "_troe" if reaction_type == "troe" else \
                     "_third" if reaction_type == "third_body" else "plog"
                     
        coef_name = f"eff_fac{type_suffix}_h"
        lines = []
        curr_line = ""
        for c in coef_array:
            curr_line, lines = append_new_str(f"{c:21.15E},".replace("E","D"), curr_line, lines)
        if(len(curr_line) > 0):
            lines.append(curr_line[:-1]+"&\n")
        new_lines = add_new_array("real", coef_name, len(coef_array), lines, new_lines)
    
    return new_lines

def write_fcent_coefficients(chem: chemistry, parallel_level=1, nreact_per_block=None):
    if parallel_level > 1:
        assert nreact_per_block is not None
    
    new_lines = []
    reaction_type  = 'troe'
    coef_array = []
    reactions = chem.get_reactions_by_type(reaction_type)
    if(len(reactions.keys())==0):
        return new_lines
    if nreact_per_block is not None:    
        if len(reactions)%nreact_per_block != 0:
            for i in range(0,nreact_per_block - len(reactions)%nreact_per_block):
                chem.add_dummy_reaction(reaction_type)
    reactions = chem.get_reactions_by_type(reaction_type)
    for reaction in reactions.values():
        falloff_coeffs = reaction["troe"]["troe"] #(a,T***,T*,T**(optional))
        coef_array.extend([1.0-falloff_coeffs[0],falloff_coeffs[1],falloff_coeffs[0],falloff_coeffs[2]])
        if(len(falloff_coeffs)>3):
            coef_array.extend([1.0,falloff_coeffs[3]])
        else:
            coef_array.extend([0.0,0.0])
                     
    lines = []
    curr_line = ""
    for c in coef_array:
        curr_line, lines = append_new_str(f"{c:21.15E},".replace("E","D"), curr_line, lines)
    if(len(curr_line) > 0):
        lines.append(curr_line[:-1]+"&\n")
    new_lines = add_new_array("real", "fcent_coef_troe_h" , len(coef_array), lines, new_lines)
    
    return new_lines


def write_rocblas_coefficients(chem: chemistry, nreact_per_block):
    
    new_lines = []
    r_dict = chem.reactions

    ##pad the mechanism
    for reaction_type in ['standard', 'troe', 'third_body', 'plog']:
        reactions = chem.get_reactions_by_type(reaction_type)
        if len(reactions)%nreact_per_block != 0:
            for i in range(0,nreact_per_block - len(reactions)%nreact_per_block):
                chem.add_dummy_reaction(reaction_type)
    
    for reaction_type in ['standard', 'troe', 'third_body', 'plog']:
        reactions = chem.get_reactions_by_type(reaction_type)
        if(chem.get_num_reactions_by_type(reaction_type)==0):
            continue
        sk_coef = np.zeros((chem.n_species_sk*chem.get_num_reactions_by_type(reaction_type),))
        coef_r = np.zeros((chem.n_species_red*chem.get_num_reactions_by_type(reaction_type),))
        coef_p = np.zeros((chem.n_species_red*chem.get_num_reactions_by_type(reaction_type),))
        wdot_coef = np.zeros((chem.n_species_red*chem.get_num_reactions_by_type(reaction_type),))
        for rnum,reaction in enumerate(reactions.values()):
            # Handle sk_coef
            for species_dict, sign in [(reaction['reacts'], -1), (reaction['prods'], 1)]:
                for s in species_dict:
                    sk_coef[rnum*chem.n_species_sk + chem.stoi[s]] += sign*species_dict[s]

            # Handle wdot_coef
            ##this is opposite of others
            for species_dict, sign in [(reaction['reacts'], -1), (reaction['prods'], 1)]:
                for s in species_dict:
                    wdot_coef[rnum + chem.stoi_red[s]*len(reactions)] \
                    += sign*species_dict[s]

            # Handle coef_r
            for s in reaction['reacts']:
                coef_r[rnum*chem.n_species_red + chem.stoi_red[s]] += reaction['reacts'][s]

            # Handle coef_p
            for s in reaction['prods']:
                coef_p[rnum*chem.n_species_red + chem.stoi_red[s]] += reaction['prods'][s]

        type_suffix = "" if reaction_type == "standard" else \
                     "_troe" if reaction_type == "troe" else \
                     "_third" if reaction_type == "third_body" else "_plog"
        for coef_name, coef_array in [(f"sk_coef{type_suffix}_h", sk_coef), 
                                      (f"coef_r{type_suffix}_h", coef_r), 
                                      (f"coef_p{type_suffix}_h", coef_p), 
                                      (f"wdot_coef{type_suffix}_h", wdot_coef)]:
            lines = []
            curr_line = ""
            for c in coef_array:
                curr_line, lines = append_new_str(f"{c:.1f},", curr_line, lines)
            if(len(curr_line) > 0):
                lines.append(curr_line[:-1]+"&\n")
            new_lines = add_new_array("real", coef_name, len(coef_array), lines, new_lines)
    
    return new_lines

def write_constants_header(dirname, chem: chemistry,parallel_level=1,veclen=None,nreact_per_block=None):
    assert parallel_level > 0 and parallel_level < 5
    assert veclen is not None
    if parallel_level > 1:
        assert nreact_per_block is not None
        assert veclen is not None
    
    header_content = get_header_content(chem,
                                        parallel_level,
                                        veclen,
                                        nreact_per_block,
                                        rocblas = parallel_level==4)
    
    with open(f"{dirname}/constants_v{parallel_level}.h", "w") as f:
        f.write(header_content)
    
    if(parallel_level > 2 and parallel_level != 4):
        print("Used default value of NSP_PER_BLOCK")
        nsp_per_block = calculate_possible_nsp_per_block(chem,
                                                         nreact_per_block,
                                                         veclen,
                                                         threads_per_block_ulimit=1024)
        print(f"Possible values of NSP_PER_BLOCK: {nsp_per_block}")
    
    max_lds_size = 64*1024

    if(parallel_level==1):
        lds_usage = chem.n_species_sk*veclen*8 + chem.n_species_red*veclen*8
        if(lds_usage > max_lds_size):
            print(f"WARNING: lds_size exceedes the max lds size")
            print(f"parallel_level:{parallel_level}, veclen:{veclen}")
    elif(parallel_level == 2):
        lds_usage = chem.n_species_sk*veclen*8 + chem.n_species_red*veclen*8*2
        if(lds_usage > max_lds_size):
            print(f"WARNING: lds_size exceedes the max lds size")
            print(f"parallel_level:{parallel_level}, veclen:{veclen}, nreact_per_block:{nreact_per_block}")
    elif(parallel_level == 3):
        for n in nsp_per_block:
            lds_usage = chem.n_species_sk*veclen*8 + chem.n_species_red*veclen*8*2 + n*veclen*nreact_per_block*8
            if(lds_usage > max_lds_size):
                print(f"WARNING: lds_size exceedes the max lds size")
                print(f"parallel_level:{parallel_level}, veclen:{veclen}, nreact_per_block:{nreact_per_block}, nsp_per_block:{n}")

    
    # with open(f"{dirname}/copy_constants.cpp", "w") as f:
    #     f.write(COPY_CONSTANTS_TO_DEVICE_FUNC)
    return

def write_coef_module(dirname, chem: chemistry,parallel_level=1,nreact_per_block=None):
    assert parallel_level > 0 and parallel_level < 5
    if parallel_level > 1:
        assert nreact_per_block is not None
    
    with open(f"{dirname}/coef_m_v{parallel_level}.f90", "w") as f:
        f.write(f"module coef_m_v{parallel_level}\n")
        f.write("  implicit none\n\n")
        
        # Write Arrhenius constants
        arrhenius_lines = write_arrhenius_constants(chem,parallel_level,nreact_per_block)
        f.writelines(arrhenius_lines)
        
        #v4 version doesn't need maps it just uses rocblas without any transformation
        if(parallel_level != 4):
            # Write maps
            map_lines = write_maps(chem,parallel_level,nreact_per_block)
            f.writelines(map_lines)
        
        # Write coefficients
        if parallel_level == 4:
            coef_lines = write_rocblas_coefficients(chem,nreact_per_block)
        else:
            coef_lines = write_coefficients(chem,parallel_level,nreact_per_block)
        f.writelines(coef_lines)

        coef_lines = write_third_body_efficiencies(chem,parallel_level,nreact_per_block)
        f.writelines(coef_lines)

        coef_lines = write_fcent_coefficients(chem,parallel_level,nreact_per_block)
        f.writelines(coef_lines)

        f.writelines(write_molwts(chem,parallel_level,nreact_per_block))
        f.writelines(write_thermo_data(chem,parallel_level,nreact_per_block))
        
        f.write(f"end module coef_m_v{parallel_level}\n")
    
    return