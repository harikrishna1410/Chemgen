import re
import sys
import numpy as np
import torch
import sympy as sp
import cantera as ct
import os
import subprocess


"""
    a ckparser class that has some simple parser functions also uses some 
    open source ck parsers. When using open source parsers this will just act 
    as a wrapper around 
    actual one    
"""

class ckparser:
    def __init__(self,parser="cantera"):
        ###
        if(parser=="cantera"):
            self.parse_reactions = self.__cantera_reaction_parser
            self.parse_species = self.__cantera_species_parser
            self.parse_thermo = self.__cantera_thermo_parser
        elif(parser=="inhouse"):
            print("WARNING: inhouse parser is deprecated and may not work with all mechanisms. Use Cantera parser instead.")
            self.parse_reactions = self.__inhouse_reaction_parser
            self.parse_species = self.__inhouse_species_parser
            self.parse_thermo = self.__inhouse_thermo_parser
        else:
            raise ValueError("parser not found")
        self.__elem_wt = {}
        self.__elem_wt["H"] = 1.007969975471497E0
        self.__elem_wt["O"] = 1.599940013885498E1
        self.__elem_wt["C"] = 1.201115036010742E1
        self.__elem_wt["N"] = 1.400669956207276E1

        ##supported reaction types
        self.__reaction_types = ["standard","troe","third_body"]
        ##cantera reaction types to my reaction types
        self.__ct_to_ckp_type = {
                                "falloff-Troe": "troe",
                                "three-body-Arrhenius": "third_body",
                                "Arrhenius": "standard"
                                }
    @property
    def reaction_types(self):
        return self.__reaction_types
    
    def __cantera_reaction_parser(self, ck_file, therm_file=None):
        import cantera as ct
        import subprocess
        import os

        # Check if YAML file already exists
        yaml_file = os.path.splitext(ck_file)[0] + '.yaml'
        if not os.path.exists(yaml_file):
            # Convert Chemkin files to YAML using ck2yaml
            if(therm_file is not None):
                subprocess.run(['ck2yaml', '--input', ck_file, '--thermo', therm_file, '--output', yaml_file], check=True)
            else:
                subprocess.run(['ck2yaml', '--input', ck_file, '--output', yaml_file], check=True)
        
        gas = ct.Solution(yaml_file)
        reactions = gas.reactions()
        r_dict = {}
        idx = 0
        for r in reactions:
            r_dict[idx] = {
                "eqn": r.equation,
                "reacts": dict(r.reactants),
                "prods": dict(r.products),
            }
            r_dict[idx]["type"] = self.__ct_to_ckp_type.get(r.reaction_type, "unknown")
            r_dict[idx]["ct"] = r #cantera reaction object
            r_dict[idx]["dup"] = r.duplicate
            r_dict[idx]["reversible"] = r.reversible
            if r_dict[idx]["type"] == "troe":
                r_dict[idx]["arh"] = (r.rate.high_rate.pre_exponential_factor, 
                                      r.rate.high_rate.temperature_exponent, 
                                      r.rate.high_rate.activation_energy)
                r_dict[idx]["troe"] = {
                    "low": (r.rate.low_rate.pre_exponential_factor,
                            r.rate.low_rate.temperature_exponent,
                            r.rate.low_rate.activation_energy),
                    "troe": tuple(r.rate.falloff_coeffs)
                }
            elif r_dict[idx]["type"] == "third_body":
                r_dict[idx]["arh"] = (r.rate.pre_exponential_factor, 
                                      r.rate.temperature_exponent, 
                                      r.rate.activation_energy)
                r_dict[idx]["third_body"] = {sp: eff - 1.0 for sp, eff in r.efficiencies.items() if eff != 1.0}
            elif r_dict[idx]["type"] == "standard":
                r_dict[idx]["arh"] = (r.rate.pre_exponential_factor, 
                                      r.rate.temperature_exponent, 
                                      r.rate.activation_energy)
            else:
                continue

            idx += 1
            
        return r_dict
        
    def __cantera_species_parser(self, ck_file):
        yaml_file = os.path.splitext(ck_file)[0] + '.yaml'
        if not os.path.exists(yaml_file):
            subprocess.run(['ck2yaml', '--input', ck_file, '--output', yaml_file], check=True)
        
        gas = ct.Solution(yaml_file)
        return [sp.name for sp in gas.species()]

    def __cantera_thermo_parser(self, ck_file, therm_file=""):
        yaml_file = os.path.splitext(ck_file)[0] + '.yaml'
        if not os.path.exists(yaml_file):
            if(therm_file == ""):
                subprocess.run(['ck2yaml', '--input', ck_file, '--output', yaml_file], check=True)
            else:
                subprocess.run(['ck2yaml', '--input', ck_file, '--thermo', therm_file, '--output', yaml_file], check=True)
        
        gas = ct.Solution(yaml_file)
        thermo_data = {}
        for sp in gas.species():
            nasa_coeffs = sp.thermo.coeffs
            temp_limits = [sp.thermo.min_temp, sp.thermo.max_temp, sp.thermo.coeffs[0]]
            thermo_data[sp.name] = (temp_limits, nasa_coeffs)
        return thermo_data
        
    """
        fuction removes all the comment lines between two lines. the lines
        have to match the start and end exactly
    """
    def __strip_parts(self,lines,start,end):
        lower_lines = [l.lower().strip() for l in lines if l != "" and l[0] != "!"]
        start_idx =[idx for idx,l in enumerate(lower_lines) if start == l][0]
        end_idx =lower_lines[start_idx:].index(end.lower())
        return lower_lines[start_idx+1:start_idx + end_idx]
    """
        function to parse all the reactions
        order_change = True puts all standard,falloff,third-body,plog reactions
        in that order
    """
    
    def __inhouse_reaction_parser(self, ck_file, therm_file):
        all_lines=open(ck_file,"r").read().splitlines()
        
        ###
        react_lines = self.__strip_parts(all_lines,"reactions","end")

        ###I simply used dict of dicts
        ###          rnum
        ##        /   |   \
        ###      eqn reacts prods arh  troe plog third-body
        r_dict = {}
        rnum = 0
        plog_r = []
        troe_r = []
        third_r = []
        for l in react_lines:
            if("!" in l):
                l = l.split("!")[0]
            if("=" in l):
                rnum = rnum + 1
                r_dict[rnum] = {}
                temp_list = [i for i in l[::-1].strip().replace("\t"," ").split(" ") if i!=""]
                r_dict[rnum]["eqn"] = "".join(temp_list[3:])[::-1]
                r_dict[rnum]["arh"] = tuple([float(i[::-1]) for i in temp_list[:3][::-1]])

                r_dict[rnum]["reacts"],r_dict[rnum]["prods"] = \
                    self.__parse_reaction_eqn(r_dict[rnum]["eqn"], ck_file)
            ### I am assuming either troe or plog
            ### I am just adding these as strings when needed I will use these
            ### I am lucky that NNH reactions for H2 JICF doesn't have any of these!!!
            elif("low" in l):
                r_dict[rnum]["troe"] = {}
                temp_list = [i for i in l.strip().replace("\t"," ").split(" ") if i!=""]
                r_dict[rnum]["troe"]["low"] = tuple([float(i) for i in temp_list[1:-1]])
            elif("troe" in l):
                temp_list = [i for i in l.strip().replace("\t"," ").split(" ") if i!=""]
                r_dict[rnum]["troe"]["troe"] = tuple([float(i) for i in temp_list[1:-1]])
                troe_r.append(rnum)
            elif("plog" in l):
                if ("plog" in r_dict[rnum].keys()):
                    num = len(r_dict[rnum]["plog"].keys())
                    r_dict[rnum]["plog"][num+1] = l.strip().split()[1:-1]
                else:
                    r_dict[rnum]["plog"] = {}
                    r_dict[rnum]["plog"][1] = l.strip().split()[1:-1]
                    plog_r.append(rnum)
            elif("dup" in l):
                r_dict[rnum]["dup"] = True
            ###I am just assuming this will be a third body collision efficiency line
            else:
                r_dict[rnum]["third_body"] = {}
                temp_list = [i.strip() for i in l.split("/")[:-1]]
                for i in range(len(temp_list)//2):
                    if(float(temp_list[2*i+1])-1.0 != 0.0):
                        r_dict[rnum]["third_body"][temp_list[2*i]] = float(temp_list[2*i+1])-1.0
                if(rnum not in troe_r):
                    third_r.append(rnum)
    ###
        return r_dict
        
        
    """
        returns all the species in .inp chemkin file
    """
    
    def __inhouse_species_parser(self, ck_file):
        f=open(ck_file)
        lines=f.readlines()
        f.close()
        ## according to chemkin 2000 manual species can be written as 
        ####SPECIES ........................... END
        ####SPEC ........................... END
        ####
        spec_lines = []
        ###first get the space seperated species lines 
        in_spec = False
        for l in lines:
            ###comment line
            if(l.strip().split("!")[0] == ""):
                continue
            if("REACTIONS" in l):
                break
            if ("SPEC" in l or "SPECIES" in l):
                in_spec=True
                ###get line after the species 
                temp_line = l.replace("SPECIES","").replace("SPEC","")
                ###remove the commented lines
                temp_line = temp_line.split("!")[0]
                spec_lines.append(temp_line)
                if("END" in temp_line):
                    break
            else:
                if(in_spec):
                    if("END" in l):
                        break
                    else:
                        spec_lines.append(l.split("!")[0])
        ###now just remove white spaces from all the lines
        spec_list = []
        for l in spec_lines:
            temp_list = l.strip().replace("\t"," ").split(" ")
            temp_list = [i for i in temp_list if (i != "")]
            spec_list = spec_list + temp_list
        spec_list = [i.strip() for i in spec_list]
        return [i.lower() for i in spec_list]
    
    """
        function to parse the thermodynamic data
    """

    def __inhouse_thermo_parser(self, ck_file, therm_file):
        species = self.__inhouse_species_parser(ck_file)
        thermo_data = {}
        for spec in species:
            thermo_data[spec] = self.__get_nasa_coffiecients(spec, therm_file)
        return thermo_data
    
    """
        function returns the nasa polynomila coeff and temeparture limits
    """
    
    def __get_nasa_coffiecients(self, spec_name, therm_file):
        coef = []
        temp_limits = []
        with open(therm_file, "r") as filestream:
            count = 0
            flag_F = False
            for line in filestream:
                l = line.strip('\n').split("!")[0].split()
                l = [i for i in l if i!=""]
                if(len(l) == 0):
                    continue
                if (l[0].strip() == spec_name):
                    temp_limits = temp_limits + [float(i) for i in l[-4:-1]]
                    flag_F = True
                    count = 0
                count += 1
                if (count > 4 and flag_F):
                    break
                if flag_F and count > 1:
                    for i in range (0,75,15):
                        coef.append(np.float64(l[0][i:i+15]))
        
        return temp_limits,coef

    """
        given the reaction name returns the products and reactants
        also returns its stoichiometric coefficients
    """
    
    def __parse_reaction_eqn(self, rname, ck_file, remove_dups=False):
        if(rname == ""):
            return {},{}
##
        if("<=>" in rname):
            sym = "<=>"
        elif("=>" in rname):
            sym = "=>"
        else:
            sym="="
        ###first remove the third body
        rname = rname.replace("(+m)","")
        rname = rname.replace("+m","")
        ###
        reacts_temp = [i.strip() for i in rname.split(sym)[0].split("+")]
        prods_temp =  [i.strip() for i in rname.split(sym)[1].split("+")]
        ##
        all_specs = self.__inhouse_species_parser(ck_file)
        reacts = {}
        prods = {}
        while (len(reacts_temp)>0):
            for spec in all_specs:
                if(spec == reacts_temp[0]):
                    if(spec not in reacts.keys()):
                        reacts[spec] = 1
                    else:
                        reacts[spec] = reacts[spec] + 1
                    reacts_temp.pop(0)
                    break
                elif(re.fullmatch("[0-9]*%s"%(spec),reacts_temp[0]) is not None):
                    if(spec not in reacts.keys()):
                        reacts[spec] = int(reacts_temp[0].replace(spec,""))
                    else:
                        reacts[spec] = reacts[spec] + int(reacts_temp[0].replace(spec,""))
                    reacts_temp.pop(0)
                    break
                if(spec==all_specs[-1]):
                    print("Error: %s is not a valid species"%(reacts_temp[0]))
                    sys.exit()
        ##products
        while (len(prods_temp)>0):
            for spec in all_specs:
                if(spec == prods_temp[0]):
                    if(spec not in prods.keys()):
                        prods[spec] = 1
                    else:
                        prods[spec] = prods[spec] + 1
                    prods_temp.pop(0)
                    break
                elif(re.fullmatch("[0-9]*%s"%(spec),prods_temp[0]) is not None):
                    if(spec not in prods.keys()):
                        prods[spec] = int(prods_temp[0].replace(spec,""))
                    else:
                        prods[spec] = prods[spec] + int(prods_temp[0].replace(spec,""))
                    prods_temp.pop(0)
                    break
                if(spec==all_specs[-1]):
                    print("Error: %s is not a valid species"%(prods_temp[0]))
                    sys.exit()
        ###now check if reacts and prods have same species
        if(remove_dups):
            rkeys = list(reacts.keys())
            pkeys = list(prods.keys())
            for s in rkeys:
                if(s in pkeys):
                    if(reacts[s] > prods[s]):
                        reacts[s] = reacts[s] - prods[s]
                        prods.pop(s)
                    elif(prods[s] > reacts[s]):
                        prods[s] = prods[s] - reacts[s]
                        reacts.pop(s)
                    else:
                        reacts.pop(s)
                        prods.pop(s)
        return reacts,prods

    """ 
        function to parse the elements 
    """
    def __parse_element_data(self, ck_file):
        all_lines = open(ck_file,"r").read().splitlines()

        elem_line = self.__strip_parts(all_lines,"elements","end")
        elements = [i for l in elem_line for i in l.strip().replace("\t"," ").split(" ")]
        return elements


