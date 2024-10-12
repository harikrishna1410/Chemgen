from .parser import ckparser
import cantera as ct
import networkx as nx
import sympy as sp
import math
import numpy as np
import re

"""
    Class to hold the info on the whole chemistry and perform manupulations on it
    NOTE: doesn't hold the compute graph. its done by the chemistry_nn
    Options:
        parser: A ck parser object        
        build_graph: builds a non directional graph of the whole chemistry
        TODO:
        build_expr: builds the expressions for each and every reaction
        identify qssa species
"""

class chemistry:
    def __init__(self, 
                ck_file, 
                parser, 
                therm_file=None, 
                build_expr=False, #builds the sympy expressions for the RF and RB
                build_graph=False,
                qssa_species=[]):
        
        assert isinstance(parser, ckparser)

        self.__parser = parser
        self.__reactions_dict = parser.parse_reactions(ck_file, therm_file)
        self.n_reactions = max(self.__reactions_dict.keys())
        self.__species_list,self.species_dict = parser.parse_species(ck_file)
        self.n_species_sk = len(self.__species_list)
        self.__qssa_species = qssa_species
        self.__reduced_species_list = [i for i in self.__species_list if i not in self.__qssa_species]
        self.n_species_red = len(self.__reduced_species_list)
        self.__thermo_data = parser.parse_thermo(ck_file, therm_file)
        ##create a specname to idx map for the full mechanism
        self.__stoi = {s:i for i,s in enumerate(self.__species_list)}
        ##create an idx to specname map for the full mechanism
        self.__itos = {i:s for i,s in enumerate(self.__species_list)}
        ##create a specname to idx map for the reduced mechanism
        self.__stoi_red = {s:i for i,s in enumerate(self.__reduced_species_list)}
        ##create an idx to specname map for the reduced mechanism
        self.__itos_red = {i:s for i,s in enumerate(self.__reduced_species_list)}
        ##create chemistry graph
        self.__chemistry_graph = None
        if(build_graph):
            self.create_chemistry_graph()
        # Initialize reaction types dictionary
        self.__reaction_types = parser.reaction_types

        self.__reaction_types_list = {}
        for rtype in self.__reaction_types:
            self.__reaction_types_list[rtype] = []
        for rnum in self.__reactions_dict.keys():
            self.__reaction_types_list[self.__reactions_dict[rnum]["type"]].append(rnum)

    @property
    def constants(self):
        return self.__constants

    @property 
    def reactions(self):
        return self.__reactions_dict
    
    @property
    def species(self):
        return self.__species_list

    @property
    def reduced_species(self):
        return self.__reduced_species_list

    @property
    def qssa_species(self):
        return self.__qssa_species

    def is_qssa(self, reaction_number):
        if self.__qssa_species is None:
            return False
        r_dict = self.__reactions_dict[reaction_number]
        reacts = list(r_dict["reacts"].keys())
        prods = list(r_dict["prods"].keys())
        return bool(set(self.__qssa_species) & set(reacts + prods))

    def get_num_reactions_by_type(self,reaction_type):
        return len(self.__reaction_types_list.get(reaction_type,[]))
    
    def get_reactions_by_type(self,reaction_type):
        return {rnum: self.__reactions_dict[rnum] for rnum in self.__reaction_types_list.get(reaction_type, [])}

    def species_index(self,species_name):
        return self.__stoi[species_name]

    def species_name(self,species_idx):
        return self.__itos[species_idx]

    def reduced_species_index(self,species_name):
        return self.__stoi_red[species_name]

    def reduced_species_name(self,species_idx):
        return self.__itos_red[species_idx]
    
    def find_max_specs(self,good_number=False):
        max_sp = 0
        for r_dict in self.reactions.values():
            reacts_dict, prods_dict = r_dict["reacts"], r_dict["prods"]
            n_sp = max(len(reacts_dict.keys())  , len(prods_dict.keys()))
            max_sp = max(max_sp, n_sp)
        if good_number:
            return 2**math.ceil(math.log2(max_sp))
        else:
            return max_sp

    def find_max_third_body(self):
        max_reacts = 0
        for r_dict in self.get_reactions_by_type("third_body").values():
            if "third_body" in r_dict:
                reacts_dict = r_dict["third_body"]
                max_reacts = max(max_reacts, len(reacts_dict.keys()))
        return max_reacts

    def add_dummy_reaction(self, reaction_type="standard"):
        """
        Add a dummy reaction with all zero coefficients and a dummy equation
        that uses the first species on left and right hand side. It's also reversible.
        
        Args:
            reaction_type (str): The type of reaction to add. Defaults to "standard".
        
        Returns:
            int: The number of the newly added dummy reaction.
        """
        dummy_reaction_number = max(self.__reactions_dict.keys()) + 1
        first_species = self.__species_list[0]
        
        dummy_reaction = {
            "type": reaction_type,
            "eqn": f"{first_species} <=> {first_species}",
            "reacts": {first_species: 0.0},
            "prods": {first_species: 0.0},
            "arh": (1.0, 0.0, 0.0),  # [A, beta, Ea] all set to zero
            "reversible": True,
            "ct":None,
            "dup": False
        }
        
        # Add additional fields based on reaction type
        if reaction_type == "troe":
            dummy_reaction["troe"] = {
                "low": (0.0, 0.0, 0.0),
                "troe": (0.0, 0.0, 0.0, 0.0)
            }
        elif reaction_type == "third_body":
            dummy_reaction["third_body"] = {first_species: 0.0}
        elif reaction_type == "plog":
            dummy_reaction["plog"] = {1: [0.0, 0.0, 0.0, 0.0]}  # [P, A, beta, Ea]
        
        self.__reactions_dict[dummy_reaction_number] = dummy_reaction
        self.__reaction_types_list[reaction_type].append(dummy_reaction_number)
        
        self.n_reactions += 1
        return dummy_reaction_number

class chemistry_expressions:
    def __init__(self, chem, vec=False, language='python'):
        self.chem = chem        
        self.reaction_expressions = {}
        self.ytoc_expr = []
        self.exp_g_expr = {}
        self.language = language.lower()
        if self.language not in ['python', 'fortran']:
            raise ValueError("Language must be either 'python' or 'fortran'")

        self.create_expressions()
        self.vec = vec
        self.Rc = 1.9872155832  # cal/(mol·K)
        self.R0 = 8.314510e+07
        self.Patm = 1013250.0
        if self.vec:
            self.vectorize_expressions()

    def create_expressions(self):
        for reaction_number, reaction in self.chem.reactions.items():
            self.reaction_expressions[reaction_number] = self.create_reaction_expression(reaction_number, reaction)
        self.create_ytoc_expr()
        self.create_exp_g_expr()

    def create_exp_g_expr(self):
        for sp_name in self.chem.species:
            thermo = self.chem.species_dict[sp_name].input_data["thermo"]
            
            low = thermo["data"][0]
            high = thermo["data"][1]
            if(thermo["model"] != "NASA7"):
                raise ValueError("thermo model != NASA7")
            temp_range = thermo["temperature-ranges"]

            self.exp_g_expr[sp_name] = [f"if T < {temp_range[1]}:"]
            # s-h expression from chemkin low polynomial
            smh_low = [
                f"    {low[6]-low[0]:.15e}\\",
                f"    {'+' if low[0] >= 0 else '-'} {abs(low[0]):.15e} * np.log(T)\\",
                f"    {'+' if low[1] >= 0 else '-'} {abs(low[1]/2.0):.15e} * T\\",
                f"    {'+' if low[2] >= 0 else '-'} {abs(low[2]/6.0):.15e} * T**2\\",
                f"    {'+' if low[3] >= 0 else '-'} {abs(low[3]/12.0):.15e} * T**3\\",
                f"    {'+' if low[4] >= 0 else '-'} {abs(low[4]/20.0):.15e} * T**4\\",
                f"    {'-' if low[5] >= 0 else '+'} {abs(low[5]):.15e} / T"
            ]
            self.exp_g_expr[sp_name].append(f"    smh = \\")
            self.exp_g_expr[sp_name].extend(smh_low)
            self.exp_g_expr[sp_name].append("else:")
            smh_high = [
                f"    {high[6]-high[0]:.15e}\\",
                f"    {'+' if high[0] >= 0 else '-'} {abs(high[0]):.15e} * np.log(T)\\",
                f"    {'+' if high[1] >= 0 else '-'} {abs(high[1]/2.0):.15e} * T\\",
                f"    {'+' if high[2] >= 0 else '-'} {abs(high[2]/6.0):.15e} * T**2\\",
                f"    {'+' if high[3] >= 0 else '-'} {abs(high[3]/12.0):.15e} * T**3\\",
                f"    {'+' if high[4] >= 0 else '-'} {abs(high[4]/20.0):.15e} * T**4\\",
                f"    {'-' if high[5] >= 0 else '+'} {abs(high[5]):.15e} / T"
            ]
            self.exp_g_expr[sp_name].append(f"    smh = \\")
            self.exp_g_expr[sp_name].extend(smh_high)
            
            self.exp_g_expr[sp_name].append(f"EG[{self.chem.species_index(sp_name)}] = np.exp(smh)")


    def create_ytoc_expr(self):
        self.ytoc_expr = []
        for sp in self.chem.reduced_species:
            reduced_index = self.chem.reduced_species_index(sp)
            molecular_weight = self.chem.species_dict[sp].molecular_weight
            self.ytoc_expr.append(
                f"C[{reduced_index}] = Y[{reduced_index}] / "
                f"({molecular_weight:.15e})"
            )
        self.ytoc_expr.append(f"ctot = 0.0")
        self.ytoc_expr.append(f"for i in range({self.chem.n_species_red}):")
        self.ytoc_expr.append("    ctot = ctot + C[i]")
        self.ytoc_expr.append(f"for i in range({self.chem.n_species_red}):")
        self.ytoc_expr.append(f"    C[i] = C[i]*P/(R0*ctot*T)")
        if(self.chem.get_num_reactions_by_type("third_body")>0 
            or self.chem.get_num_reactions_by_type("troe")>0):
            self.ytoc_expr.append(f"ctot = 0.0")
            self.ytoc_expr.append(f"for i in range({self.chem.n_species_red}):")
            self.ytoc_expr.append(f"    ctot = ctot + C[i]")

    def create_reaction_expression(self, reaction_number, reaction):
        reactants_expr = self.create_species_expression(reaction['reacts'])
        products_expr = self.create_species_expression(reaction['prods'])
        eqk = self.create_equilibrium_expression(reaction["reacts"], reaction["prods"])
        
        expr = {}
        
        if reaction["type"] == "third_body":
            A, beta, Ea = reaction['arh']
            expr["kf"] = f"{A:.15e}" \
                    + (f" * (T ** {beta})" if abs(beta) > 0.0 else "") \
                    + f" * np.exp({-Ea:.15e} / (Rc * T))"
            expr["kb"] = f"kf * ({eqk})" if reaction['reversible'] else "0"
            expr["rr"] = f"(ctot " + ("+" if len(reaction["third_body"]) > 0 else "") \
                        + "+".join([f"{eff:.15e}*C[{self.chem.reduced_species_index(sp)}]\\\n    " for sp,eff in reaction["third_body"].items()])\
                         + f")*(kf * ({reactants_expr}) - kb * ({products_expr}))"
        elif reaction["type"] == "troe":
            A, beta, Ea = reaction['arh']
            A0, beta0, Ea0 = reaction['troe']['low']
            troe_params = reaction['troe']['troe']
            a, T3, T1 = troe_params[:3]
            T2 = troe_params[3] if len(troe_params) > 3 else 0.0
            expr["M"] = "ctot"
            if reaction["third_body"]:
                expr["M"] += " + " + " + ".join([f"{eff:.15e}*C[{self.chem.reduced_species_index(sp)}]" for sp, eff in reaction["third_body"].items()])
            expr["k0"] = f"{A0:.15e}" \
                        + (f" * (T ** {beta0})" if abs(beta0) > 0.0 else "") \
                        + f" * np.exp(-{Ea0:.15e} / (Rc * T))"
            expr["kinf"] = f"{A:.15e}" \
                    + (f" * (T ** {beta})" if abs(beta) > 0.0 else "") \
                    + f" * np.exp({-Ea:.15e} / (Rc * T))"
            expr["Pr"] = f"(k0 * M) / kinf"
            expr["Fcent"] = f"(1-{a}) * np.exp(-T/{T3}) + {a} * np.exp(-T/{T1})"
            if T2 > 0:
                expr["Fcent"] += f" + np.exp(-{T2}/T)"
            expr["C1"] = f"-0.4 - 0.67 * np.log10(Fcent)"
            expr["N"] = f"0.75 - 1.27 * np.log10(Fcent)"
            expr["F1"] = f"(np.log10(Pr) + C1) / (N - 0.14 * (np.log10(Pr) + C1))"
            expr["F"] = f"10 ** (np.log10(Fcent) / (1 + F1**2))"
            expr["kf"] = f"kinf * (Pr / (1 + Pr)) * F"
            expr["kb"] = f"kf * ({eqk})" if reaction['reversible'] else "0"
            expr["rr"] = f"kf * ({reactants_expr}) - kb * ({products_expr})"
        elif reaction["type"] == "standard":    
            A, beta, Ea = reaction['arh']
            expr["kf"] = f"{A:.15e}" \
                    + (f" * (T ** {beta})" if abs(beta) > 0.0 else "") \
                    + f" * np.exp({-Ea:.15e} / (Rc * T))"
            expr["kb"] = f"kf * ({eqk})" if reaction['reversible'] else "0"
            expr["rr"] = f"kf * ({reactants_expr}) - kb * ({products_expr})"
        elif reaction["type"] == "plog":
            # First pass to identify and sum duplicate pressure points
            pressure_points = {}
            for c in reaction['plog']:
                if c[0] not in pressure_points:
                    pressure_points[c[0]] = []
                pressure_points[c[0]].append(c)

            # Build kf strings for each unique pressure point
            kf_strings = {}
            for pressure, conditions in pressure_points.items():
                kf_sum = " + ".join([f"{c[1]:.15e}" + (f" * (T ** {c[2]})" if abs(c[2]) > 0.0 else "") + f" * np.exp({-c[3]:.15e} / (Rc * T))" for c in conditions])
                kf_strings[pressure] = kf_sum

            # Sort pressure points
            sorted_pressures = sorted(pressure_points.keys())

            # Build expressions using the summed kf strings
            expr = {}
            for i, p in enumerate(sorted_pressures):
                if i == 0:
                    expr[f"if P < {p:.15e}:"] = [f"if P < {p:.15e}:"]
                    expr[f"if P < {p:.15e}:"].append(f"    kfl={kf_strings[p]}")
                    expr[f"if P < {p:.15e}:"].append(f"    kfh={kf_strings[p]}")
                    expr[f"if P < {p:.15e}:"].append(f"    logPl=1.0")
                    expr[f"if P < {p:.15e}:"].append(f"    logPh={np.log(p):.15e}")
                elif i < len(sorted_pressures) - 1:
                    expr[f"elif P < {sorted_pressures[i+1]:.15e}:"] = [f"elif P < {sorted_pressures[i+1]:.15e}:"]
                    expr[f"elif P < {sorted_pressures[i+1]:.15e}:"].append(f"    kfl={kf_strings[p]}")
                    expr[f"elif P < {sorted_pressures[i+1]:.15e}:"].append(f"    kfh={kf_strings[sorted_pressures[i+1]]}")
                    expr[f"elif P < {sorted_pressures[i+1]:.15e}:"].append(f"    logPl={np.log(p):.15e}")
                    expr[f"elif P < {sorted_pressures[i+1]:.15e}:"].append(f"    logPh={np.log(sorted_pressures[i+1]):.15e}")

            expr["else:"] = ["else:"]
            last_p = sorted_pressures[-1]
            expr["else:"].append(f"    kfl={kf_strings[last_p]}")
            expr["else:"].append(f"    kfh={kf_strings[last_p]}")
            expr["else:"].append(f"    logPl={np.log(last_p):.15e}")
            expr["else:"].append(f"    logPh=100.0")

            expr["kf"] = "np.exp(np.log(kfl) + (np.log(kfh)-np.log(kfl))*(np.log(P)-logPl)/(logPh-logPl))"
            expr["kb"] = f"kf * ({eqk})" if reaction['reversible'] else "0.0"
            expr["rr"] = f"kf * ({reactants_expr}) - kb * ({products_expr})"
        
        expr["wdot"] = self.get_species_production_rate(reaction_number, reaction)
        return expr

    def create_species_expression(self, species_dict):
        return " * ".join([f"C[{self.chem.reduced_species_index(species)}]" + (f"**{coeff}" if coeff > 1.0 else "")
                           for species, coeff in species_dict.items()])
    
    def create_equilibrium_expression(self, reactants_dict, products_dict):
        reactants = " * ".join([f"EG[{self.chem.species_index(species)}]" + (f"**({coeff})" if coeff > 1.0 else "")
                                for species, coeff in reactants_dict.items()])
        products = " * ".join([f"EG[{self.chem.species_index(species)}]" + (f"**{coeff}" if coeff > 1.0 else "")
                               for species, coeff in products_dict.items()])
        if(abs(sum(products_dict.values())-sum(reactants_dict.values())) > 0):
            if(sum(products_dict.values())-sum(reactants_dict.values()) == 1):
                pfac = f"* pfac"
            else:
                pfac = f"* pfac**{sum(products_dict.values())-sum(reactants_dict.values()):0.1f}"
        else:
            pfac = ""
        return f"(({reactants})/({products}{pfac}))"

    def get_reaction_expression(self, reaction_number):
        return self.reaction_expressions[reaction_number]
        
    def get_species_production_rate(self, reaction_number, reaction):
        wdot_expressions = []
        for species_name, coeff in reaction["reacts"].items():
            species_idx = self.chem.reduced_species_index(species_name)
            wdot_expr = f"wdot[{species_idx}] = wdot[{species_idx}] " + (f"- {coeff:.2f} * rr" if coeff > 1.0 else "- rr")
            wdot_expressions.append(wdot_expr)
        
        for species_name, coeff in reaction["prods"].items():
            species_idx = self.chem.reduced_species_index(species_name)
            wdot_expr = f"wdot[{species_idx}] = wdot[{species_idx}] " + (f"+ {coeff:.2f} * rr" if coeff > 1.0 else "+ rr")
            wdot_expressions.append(wdot_expr)
        
        return wdot_expressions

    def vectorize_expressions(self):
        vector_vars = ['T', 'C', 'EG', 'kf', 'kb', 'rr', 'wdot', 'Y', "ctot", "P"]

        # Vectorize reaction expressions
        for reaction_number, reaction_expr in self.reaction_expressions.items():
            vectorized_expr = {}
            
            if self.chem.reactions[reaction_number]['type'] == 'plog':
                vectorized_expr = self.vectorize_plog_reaction(reaction_expr, vector_vars)
            else:
                for key, expr in reaction_expr.items():
                    if key == "wdot":
                        vectorized = [self.vectorize_wdot_expr(expr) for expr in expr]
                    else:
                        vectorized = expr
                        for var in vector_vars:
                            vec_var = f"{var}"
                            vectorized = vectorized.replace(f"{var}[", f"{vec_var}[:,")
                    
                    vectorized_expr[key] = vectorized
            
            self.reaction_expressions[reaction_number] = vectorized_expr

        # Vectorize ytoc expressions
        vectorized_ytoc = []
        for expr in self.ytoc_expr:
            vectorized = expr
            for var in vector_vars:
                vec_var = f"{var}"
                vectorized = vectorized.replace(f"{var}[", f"{vec_var}[:,")
            vectorized_ytoc.append(vectorized)
        self.ytoc_expr = vectorized_ytoc

        # Vectorize exp_g expressions
        for sp_name in self.chem.species:
            vectorized_exp_g = ["for i in range(veclen):"]
            for expr in self.exp_g_expr[sp_name]:
                indented_expr = "    " + expr
                indented_expr = indented_expr.replace("T", "T[i]")
                indented_expr = re.sub(r'EG\[(\d+)\]', r'EG[i,\1]', indented_expr)
                vectorized_exp_g.append(indented_expr)
            self.exp_g_expr[sp_name] = vectorized_exp_g

    def vectorize_plog_reaction(self, reaction_expr, vector_vars):
        vectorized_expr = {}
        
        for key, expr in reaction_expr.items():
            if key.startswith("if"):
                vectorized = ["for i in range(veclen):"]
            else:
                vectorized = []
            if key.startswith("if") or key.startswith("elif") or key == "else:":
                for line in expr:
                    vectorized_line = "    " + line
                    for var in ["T"]:
                        vectorized_line = vectorized_line.replace(f"{var}", f"{var}[i]")
                    vectorized.append(vectorized_line)
                vectorized[0] = vectorized[0].replace("P","P[i]")
                vectorized[1] = vectorized[1].replace("P","P[i]")
                vectorized_expr[key] = vectorized
            elif key == "kf":
                vectorized_line = expr
                vectorized_line = vectorized_line.replace("logPl","logkl")\
                               .replace("logPh","logkh")\
                               .replace("kfl","bfl")\
                               .replace("kfh","bfh")
                for var in vector_vars:
                    vectorized_line = vectorized_line.replace(f"{var}", f"{var}[i]")
                vectorized_line = vectorized_line.replace("logkl","logPl")\
                               .replace("logkh","logPh")\
                               .replace("bfl","kfl")\
                               .replace("bfh","kfh")
                vectorized_expr[key] = vectorized_line
            elif key == "wdot":
                vectorized = [self.vectorize_wdot_expr(expr) for expr in expr]
                vectorized_expr[key] = vectorized
            else:
                vectorized = expr
                for var in vector_vars:
                    vec_var = f"{var}"
                    vectorized = vectorized.replace(f"{var}[", f"{vec_var}[:,")
                vectorized_expr[key] = vectorized
        
        return vectorized_expr

    def vectorize_wdot_expr(self, expr):
        return expr.replace("wdot[", "wdot[:,").replace("rr", "rr[:]")

    def format_expression(self, expr):
        if self.language == 'python':
            return expr
        elif self.language == 'fortran':
            return self.convert_to_fortran(expr)

    def convert_to_fortran(self, expr):
        if isinstance(expr, list):  # This is a block structure (if, else, elif, or for)
            first_line = expr[0].strip()
            indent = " " * (len(expr[0]) - len(expr[0].strip()))
            if first_line.startswith("for "):
                loop_parts = first_line.split()
                loop_var = loop_parts[1]
                range_parts = loop_parts[3].strip("range(").split(",")
                range_parts[-1] = range_parts[-1].strip("):")
                if len(range_parts) == 1:
                    start = "1"
                    end = range_parts[0]
                elif len(range_parts) == 2:
                    start, end = range_parts
                else:
                    start, end, step = range_parts
                    # Note: Fortran's step is handled differently, may need adjustment
                
                fortran_block = [f"{indent}do {loop_var} = {start}, {end}"]
            elif first_line.startswith("if "):
                condition = first_line[3:-1]  # Remove 'if ' and ':'
                fortran_block = [f"{indent}if ({self.convert_to_fortran(condition)}) then"]
            elif first_line.startswith("elif "):
                condition = first_line[5:-1]  # Remove 'elif ' and ':'
                fortran_block = [f"{indent}else if ({self.convert_to_fortran(condition)}) then"]
            elif first_line == "else:":
                fortran_block = [f"{indent}else"]
            else:
                raise ValueError(f"Unexpected block start: {first_line}")
            i = 1            
            while i < len(expr):
                line = expr[i]
                if (line.strip().startswith("if ") or 
                    line.strip().startswith("elif ") or 
                    line.strip().startswith("else:") or 
                    line.strip().startswith("for ")):
                    # Start of a nested block
                    print(line)
                    nested_block = [line]
                    i += 1
                    current_indent = len(line) - len(line.lstrip())
                    while i < len(expr) and len(expr[i]) - len(expr[i].lstrip()) > current_indent:
                        print(expr[i])
                        nested_block.append(expr[i])
                        i += 1
                    converted_nested_block = self.convert_to_fortran(nested_block)
                    for c in converted_nested_block:
                        print(c)
                    fortran_block.extend([l for l in converted_nested_block])
                else:
                    fortran_block.append(self.convert_to_fortran(line))
                    i += 1
            
            if first_line.startswith("for "):
                fortran_block.append(f"{indent}end do")
            elif first_line == "else:":
                fortran_block.append(f"{indent}end if")
            
            return fortran_block
        else:  # This is a single expression
            # Convert Python-style expressions to Fortran
            expr = expr.replace("**", "**")
            expr = expr.replace("[", "(").replace("]", ")")
            expr = expr.replace("np.exp", "dexp")
            expr = expr.replace(":.15e", ":.15D0")
            expr = expr.replace("\\", "&")
            expr = expr.replace("np.log", "dlog")
            expr = expr.replace("np.log10", "dlog10")
            # Handle if, elif, else, and for statements
            if expr.strip().startswith("if "):
                condition = expr.strip()[3:-1]  # Remove 'if ' and ':'
                expr = f"if ({self.convert_to_fortran(condition)}) then"
            elif expr.strip().startswith("elif "):
                condition = expr.strip()[5:-1]  # Remove 'elif ' and ':'
                expr = f"else if ({self.convert_to_fortran(condition)}) then"
            elif expr.strip() == "else:":
                expr = "else"
            elif expr.strip().startswith("for "):
                loop_parts = expr.strip().split()
                loop_var = loop_parts[1]
                range_parts = loop_parts[3].strip("range(").split(",")
                range_parts[-1] = range_parts[-1].strip("):")
                if len(range_parts) == 1:
                    start = "1"
                    end = range_parts[0]
                elif len(range_parts) == 2:
                    start, end = range_parts
                else:
                    start, end, step = range_parts
                expr = f"do {loop_var} = {start}, {end}"

            # Increase all indices between parentheses by 1 using regex
            def increment_index(match):
                index = int(match.group(1))
                return f"({index + 1}"

            expr = re.sub(r'\((\d+)', increment_index, expr)
            
            # Handle vectorized expressions with [i,index]
            def increment_i_index(match):
                index = int(match.group(1))
                return f"(i,{index + 1}"

            expr = re.sub(r'\(i,(\d+)', increment_i_index, expr)

            # Handle vectorized expressions with [:, index]
            def increment_colon_index(match):
                index = int(match.group(1))
                return f"(:,{index + 1}"

            expr = re.sub(r'\(:,(\d+)', increment_colon_index, expr)
            
            return expr

    def write_expressions_to_file(self, filename):
        with open(filename, 'w') as f:
            if self.language == 'python':
                self.write_python_header(f)
            else:  # Fortran
                self.write_fortran_header(f)
            
            # Write ytoc expressions
            f.write("    # Y to C conversion\n" if self.language == 'python' else "    ! Y to C conversion\n")
            i = 0
            while i < len(self.ytoc_expr):
                expr = self.ytoc_expr[i]
                if expr.startswith("for "):
                    # Found a loop, collect all expressions until the indentation changes
                    loop_exprs = [expr]
                    i += 1
                    while i < len(self.ytoc_expr) and self.ytoc_expr[i].startswith("    "):
                        loop_exprs.append(self.ytoc_expr[i])
                        i += 1
                    formatted_loop = self.format_expression(loop_exprs)
                    for line in formatted_loop:
                        f.write(f"    {line}\n")
                else:
                    f.write(f"    {self.format_expression(expr)}\n")
                    i += 1
            f.write("\n")
            
            # Write exp_g expressions
            f.write("    # Exponential G calculations\n" if self.language == 'python' else "    ! Exponential G calculations\n")
            for species, exp_g_expr in self.exp_g_expr.items():
                f.write(f"    # {species}\n" if self.language == 'python' else f"    ! {species}\n")
                i = 0
                while i < len(exp_g_expr):
                    expr = exp_g_expr[i]
                    if expr.startswith("for ") or \
                        expr.startswith("if ") or \
                        expr.startswith("else") or \
                        expr.startswith("elif"):
                        # Found a loop or conditional, collect all expressions until the indentation changes
                        block_exprs = [expr]
                        i += 1
                        while i < len(exp_g_expr) and exp_g_expr[i].startswith("    "):
                            block_exprs.append(exp_g_expr[i])
                            i += 1
                        formatted_block = self.format_expression(block_exprs)
                        for line in formatted_block:
                            f.write(f"    {line}\n")
                    else:
                        f.write(f"    {self.format_expression(expr)}\n")
                        i += 1
                f.write("\n")
            
            for reaction_number, reaction_expr in self.reaction_expressions.items():
                f.write(f"    {'#' if self.language == 'python' else '!'} Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                
                if self.chem.reactions[reaction_number]['type'] == 'plog':
                    self.write_plog_reaction(f, reaction_expr)
                else:
                    for expr_type in reaction_expr.keys():
                        if expr_type != "wdot":
                            f.write(f"    {expr_type} = {self.format_expression(reaction_expr[expr_type])}\n")
                for wdot_expr in reaction_expr['wdot']:
                    f.write(f"    {self.format_expression(wdot_expr)}\n")
                f.write("\n")

            if self.language == 'python':
                f.write("    return kf, kb, rr\n")
            else:  # Fortran
                f.write("end subroutine getrates\n")

        print(f"Expressions and getrates function written to {filename}")

    def write_plog_reaction(self, f, reaction_expr):
        for key, expr_list in reaction_expr.items():
            if key.startswith("if") or key.startswith("elif") or key == "else:":
                if key.startswith("if") and self.vec:
                    expr_list[0] = self.format_expression(expr_list[0])
                    formatted_block = self.format_expression(expr_list[1:])
                    formatted_block = [expr_list[0]] + formatted_block
                else:
                    formatted_block = self.format_expression(expr_list)
                for line in formatted_block:
                    f.write(f"    {line}\n")
            elif key != "wdot":
                if(self.vec and key == "kf"):
                    vec_key = self.format_expression("    "+key+"[i]")
                    f.write(f"    {vec_key} = {self.format_expression(expr_list)}\n")
                    if(self.language == "fortran"):
                        f.write("    end do\n")
                else:
                    f.write(f"    {key} = {self.format_expression(expr_list)}\n")

    def write_python_header(self, f):
        f.write("import numpy as np\n\n")
        if self.vec:
            f.write("def getrates(veclen, T, Y, P, wdot):\n")
            f.write(f"    C = np.zeros((veclen, {self.chem.n_species_red}))\n")
            f.write(f"    EG = np.zeros((veclen, {self.chem.n_species_sk}))\n")
            f.write("    kf = np.zeros(veclen)\n")
            f.write("    kb = np.zeros(veclen)\n")
            f.write(f"    rr = np.zeros(veclen)\n")
            f.write(f"    ctot = np.zeros(veclen)\n")
            # Add intermediate variables for troe and plog reactions
            f.write("    k0 = np.zeros(veclen)\n")
            f.write("    kinf = np.zeros(veclen)\n")
            f.write("    Pr = np.zeros(veclen)\n")
            f.write("    Fcent = np.zeros(veclen)\n")
            f.write("    C_troe = np.zeros(veclen)\n")
            f.write("    N_troe = np.zeros(veclen)\n")
            f.write("    F1 = np.zeros(veclen)\n")
            f.write("    F = np.zeros(veclen)\n")
            f.write("    logPr = np.zeros(veclen)\n")
            f.write("    logFcent = np.zeros(veclen)\n")
            f.write("    wdot[:,:] = 0.0\n")
        else:
            f.write("def getrates(T, Y, P, wdot):\n")
            f.write(f"    C = np.zeros({self.chem.n_species_red})\n")
            f.write(f"    EG = np.zeros({self.chem.n_species_sk})\n")
            f.write("    kf = 0.0\n")
            f.write("    kb = 0.0\n")
            f.write("    rr = 0.0\n")
            f.write("    ctot = 0.0\n")
            # Add intermediate variables for troe and plog reactions
            f.write("    k0 = 0.0\n")
            f.write("    kinf = 0.0\n")
            f.write("    Pr = 0.0\n")
            f.write("    Fcent = 0.0\n")
            f.write("    C_troe = 0.0\n")
            f.write("    N_troe = 0.0\n")
            f.write("    F1 = 0.0\n")
            f.write("    F = 0.0\n")
            f.write("    logPr = 0.0\n")
            f.write("    logFcent = 0.0\n")
            f.write("    wdot[:] = 0.0\n")
        f.write(f"    Rc = {self.Rc}\n")
        f.write(f"    R0 = {self.R0}\n")
        f.write(f"    Patm = {self.Patm}\n")
        f.write("\n")
        f.write(f"    pfac = Patm/(R0*T)\n")
        

    def write_fortran_header(self, f):
        if self.vec:
            f.write("subroutine getrates(veclen, T, Y, P, wdot)\n")
            f.write("    implicit none\n")
            f.write("    integer, intent(in) :: veclen\n")
            f.write("    real(kind=8), dimension(veclen), intent(in) :: T, P\n")
            f.write(f"    real(kind=8), dimension(veclen, {self.chem.n_species_red}), intent(in) :: Y\n")
            f.write(f"    real(kind=8), dimension(veclen, {self.chem.n_species_red}), intent(out) :: wdot\n")
            f.write(f"    real(kind=8), dimension(veclen, {self.chem.n_species_red}) :: C\n")
            f.write(f"    real(kind=8), dimension(veclen, {self.chem.n_species_sk}) :: EG\n")
            f.write("    real(kind=8), dimension(veclen) :: kf, kb\n")
            f.write("    real(kind=8), dimension(veclen) :: rr, ctot, pfac\n")
            # Add intermediate variables for troe and plog reactions
            f.write("    real(kind=8), dimension(veclen) :: k0, kinf, Pr, Fcent\n")
            f.write("    real(kind=8), dimension(veclen) :: C_troe, N_troe, F1, F\n")
            f.write("    real(kind=8), dimension(veclen) :: logPr, logFcent\n")
        else:
            f.write("subroutine getrates(P, T, Y, ickwrk, rckwrk, wdot)\n")
            f.write("    implicit none\n")
            f.write("    real(kind=8), intent(in) :: T, P\n")
            f.write(f"    real(kind=8), dimension({self.chem.n_species_red}), intent(in) :: Y\n")
            f.write(f"    real(kind=8), dimension({self.chem.n_species_red}), intent(out) :: wdot\n")
            f.write(f"    real(kind=8), dimension({self.chem.n_species_red}) :: C\n")
            f.write(f"    real(kind=8), dimension({self.chem.n_species_sk}) :: EG\n")
            f.write("    real(kind=8) :: kf, kb\n")
            f.write("    real(kind=8) :: rr, ctot, pfac\n")
            # Add intermediate variables for troe and plog reactions
            f.write("    real(kind=8) :: k0, kinf, Pr, Fcent\n")
            f.write("    real(kind=8) :: C_troe, N_troe, F1, F\n")
            f.write("    real(kind=8) :: logPr, logFcent\n")
        f.write(f"    real(kind=8), parameter :: Rc = {self.Rc}\n")
        f.write(f"    real(kind=8), parameter :: R0 = {self.R0}\n")
        f.write(f"    real(kind=8), parameter :: Patm = {self.Patm}\n")
        f.write("    integer :: i\n")
        f.write("\n")
        f.write("    pfac = Patm / (R0 * T)\n")



##TODO:add these after implementing automatic qssa identification
# def validate_species_maps(self):
#         for s in self.__stoi.keys():
#             if(self.__itos[self.__stoi[s]] != s):
#                 raise ValueError("itos and stoi are not equal")
#         if(len(self.__stoi) != len(self.__species_list) or \
#             len(self.__itos) != len(self.__species_list)):
#             raise ValueError("itos/stoi don't have all the species.")
#         return
    
#     def reset_stoi_map(self,stoi):
#         self.__stoi = stoi
#         for s,i in self.__stoi.items():
#             self.__itos[i] = s
#         self.validate_species_maps()
#         return

#     def create_chemistry_graph(self):
#         self.__chem_graph = nx.MultiGraph()
#         self.__chem_graph.add_nodes_from(self.__species_list)
#         for r in self.__reactions_dict.keys():
#             rdict = self.__reactions_dict[r]
#             edges = [(i,j) for i in rdict["reacts"] for j in rdict["prods"]]
#             keys = [k for k in rdict.keys() if k not in ["reacts","prods"]]
#             add_dict = {k:rdict[k] for k in keys}
#             self.__chem_graph.add_edges_from([(i,j,add_dict) for i,j in edges])
#         return
  
#     def remap_stoi_for_qssa(self):
#         assert self.__qssa_species is not None
#         assert self.__non_qssa_species is not None
#         qssa_specs = self.__qssa_species
#         non_qssa = self.__non_qssa_species
#         stoi = {s:i for i,s in enumerate(non_qssa+qssa_specs)}
#         self.reset_stoi_map(stoi)
#         return

#     def identify_qssa(self):
#         # TODO: Implement this method
#         return

#     def init_qssa(self,qssa_specs):
#         self.__qssa_species = qssa_specs
#         self.__non_qssa_species = [s for s in self.__species_list if s not in qssa_specs]
#         return
