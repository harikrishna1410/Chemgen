from .parser import ckparser
import cantera as ct
import networkx as nx
import sympy as sp
import math
import numpy as np
import re
import os
from jinja2 import Environment,FileSystemLoader

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

    @property
    def stoi(self):
        return self.__stoi
    
    @property
    def stoi_red(self):
        return self.__stoi_red
    
    @property
    def reaction_types(self):
        return self.__reaction_types

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
                "low": (1.0, 0.0, 0.0),
                "troe": (1.0, 1.0, 1e30)
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
    def __init__(self, chem, vec=False, omp = False, mod=False,language='python'):
        self.chem = chem        
        self.reaction_expressions = {}
        self.ytoc_expr = []
        self.exp_g_expr = {}
        self.language = language.lower()
        if self.language not in ['python', 'fortran']:
            raise ValueError("Language must be either 'python' or 'fortran'")

        self.vec = vec
        self.omp = omp
        self.mod = mod
        if(self.vec and self.omp):
            raise ValueError("both vec and omp can't be true")
        
        if(self.omp and self.language == "python"):
            raise ValueError("omp only works with fortran")
        
        self.Rc = 1.987215575926745  # cal/(mol·K)
        self.R0 = 8.314510e+07
        self.Patm = 1013250.0

    def create_expressions(self,input_MW):
        for reaction_number, reaction in self.chem.reactions.items():
            self.reaction_expressions[reaction_number] = self.create_reaction_expression(reaction_number, reaction)
        self.create_ytoc_expr(input_MW)
        self.create_exp_g_expr()

    def create_exp_g_expr(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        env.globals["abs"] = abs
        if self.language == "python":
            if self.vec:
                template = env.get_template("exp_g_vec.j2")
            else:
                template = env.get_template("exp_g.j2")
        elif self.language == "fortran":
            if self.vec:
                template = env.get_template("exp_g_ftn_vec.j2")
            elif self.omp:
                template = env.get_template("reactions_omp_gpu/exp_g.j2")
            else:
                template = env.get_template("exp_g_ftn.j2")
        else:
            raise ValueError("unknown language")
        
        context = {
            "species_dict": {sp_name:self.chem.species_dict[sp_name].input_data["thermo"] for sp_name in self.chem.species}
        }
        rendered_string = template.render(context)
        self.exp_g_expr = '\n'.join('    ' + line if not line.strip().startswith('!$') else line for line in rendered_string.split('\n') if line.strip())

    def create_ytoc_expr(self,input_MW):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        if self.language == "python":
            if self.vec:
                template = env.get_template("ytoc_vec.j2")
            else:
                template = env.get_template("ytoc.j2")
        elif self.language == "fortran":
            if self.vec:
                template = env.get_template("ytoc_ftn_vec.j2")
            elif self.omp:
                template = env.get_template("reactions_omp_gpu/ytoc.j2")
            else:
                template = env.get_template("ytoc_ftn.j2")
        else:
            raise ValueError("unknown language")
        
        context = {
            "chem": {
                "nspecies": self.chem.n_species_red,
                "mw": [self.chem.species_dict[sp].molecular_weight for sp in self.chem.reduced_species],
                "has_third_body_reactions": any(r["type"] == "third_body" for r in self.chem.reactions.values()),
                "has_troe_reactions": any(r["type"] == "troe" for r in self.chem.reactions.values())
            },
            "input_MW": input_MW
        }
        rendered_string = template.render(context)
        self.ytoc_expr = '\n'.join('    ' + line if not line.strip().startswith('!$') else line for line in rendered_string.split('\n') if line.strip())

    def create_reaction_expression(self, reaction_number, reaction):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        env.globals["abs"] = abs
        env.globals["log"] = np.log

        # Calculate coeff_sum correctly
        coeff_sum = sum(reaction["prods"].values()) - sum(reaction["reacts"].values())
        context = {
                "reaction": reaction,
                "stoi": self.chem.stoi,
                "stoi_rd": self.chem.stoi_red,
                "coeff_sum": coeff_sum
        }

        if self.language == "python":
            if(self.vec):
                base_dir = "reactions_vectorised"
            else:
                base_dir = "reactions"
        elif self.language == "fortran":
            if self.vec:
                base_dir = "reactions_ftn_vec"
            elif self.omp:
                base_dir = "reactions_omp_gpu"
            else:
                base_dir = "reactions_ftn"
        else:
            raise ValueError("unknown language")
        
        if reaction["type"] == "standard":
            template = env.get_template(f"{base_dir}/standard.j2")
        elif reaction["type"] == "third_body":
            template = env.get_template(f"{base_dir}/third_body.j2")
        elif reaction["type"] == "troe":
            template = env.get_template(f"{base_dir}/troe.j2")
        elif reaction["type"] == "plog":
            if("HCN" in reaction["reacts"].keys() or "HCN" in reaction["prods"].keys()):
                print(reaction["eqn"])
                print([c[0] for c in reaction["plog"]])
            pressure_points = set()
            for c in reaction["plog"]:
                pressure_points.add(c[0])
            plog = {p:[] for p in list(pressure_points)}
            for c in reaction["plog"]:
                plog[c[0]].append(c[1:])
            context["plog"] = plog
            context["pressure_points"] = sorted(list(pressure_points))
            template = env.get_template(f"{base_dir}/plog.j2")
        
        rendered_string = template.render(context)
        expr = '\n'.join('    ' + line.rstrip() if not line.strip().startswith('!$') else line.rstrip() for line in rendered_string.split('\n') if line.strip())
        return expr

    #write the expressions to a file
    #write_rtogther: if true, write all reactions of a type together
    #input_MW: if true, the subroutine will take MW as input
    def write_expressions_to_file(self, 
                                  filename, 
                                  write_rtypes_together=False,
                                  input_MW=False):
        self.create_expressions(input_MW)
        if self.language == 'python':
            self._write_python_expressions(filename, write_rtypes_together)
        elif self.language == 'fortran':
            if self.omp:
                if self.mod:
                    self._write_chemgen_mod(filename, write_rtypes_together,input_MW)
                else:
                    self._write_omp_expressions(filename, write_rtypes_together) 
            else:
                self._write_fortran_expressions(filename, write_rtypes_together,input_MW)
        print(f"Expressions and getrates function written to {filename}")

    def _write_python_expressions(self, filename, write_rtypes_together):
        with open(filename, 'w') as f:
            self.write_python_header(f)
            
            f.write("    # Y to C conversion\n")
            f.write(self.ytoc_expr)
            f.write("\n")
            
            f.write("    # Exponential G calculations\n") 
            f.write(self.exp_g_expr)
            f.write("\n")

            if write_rtypes_together:
                for rtype in self.chem.reaction_types:
                    f.write(f"    # Reaction type: {rtype}\n")
                    for reaction_number, reaction_expr in self.reaction_expressions.items():
                        if self.chem.reactions[reaction_number]["type"] == rtype:
                            f.write(f"    # Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                            f.write(self.reaction_expressions[reaction_number])
                            f.write("\n")
            else:
                for reaction_number, reaction_expr in self.reaction_expressions.items():
                    f.write(f"    # Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                    f.write(self.reaction_expressions[reaction_number])
                    f.write("\n")

            f.write("    return kf, kb, rr\n")

    def _write_fortran_expressions(self, filename, write_rtypes_together,input_MW):
        with open(filename, 'w') as f:
            self.write_fortran_header(f,input_MW=input_MW)
            
            f.write("    ! Y to C conversion\n")
            f.write(self.ytoc_expr)
            f.write("\n")
            
            f.write("    ! Exponential G calculations\n")
            f.write(self.exp_g_expr)
            f.write("\n")

            if write_rtypes_together:
                for rtype in self.chem.reaction_types:
                    f.write(f"    ! Reaction type: {rtype}\n")
                    for reaction_number, reaction_expr in self.reaction_expressions.items():
                        if self.chem.reactions[reaction_number]["type"] == rtype:
                            f.write(f"    ! Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                            f.write(self.reaction_expressions[reaction_number])
                            f.write("\n")
            else:
                for reaction_number, reaction_expr in self.reaction_expressions.items():
                    f.write(f"    ! Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                    f.write(self.reaction_expressions[reaction_number])
                    f.write("\n")

            if self.vec:
                f.write("end subroutine getrates_i\n")
            else:
                f.write("end subroutine getrates\n")

    def _write_omp_expressions(self, filename, write_rtypes_together):
        thread_private = "private(kf,kb,rr,M,k0,kinf,Pr,Fcent,C1,N,F1,F,logPr,logFcent,smh,kfl,kfh,kbl,logPl,logPh,i,L)"
        omp_startdo = f"!$omp target teams distribute parallel do {thread_private}"
        omp_enddo = "!$omp end target teams distribute parallel do"
        startdo = "    do i = 1,veclen"
        enddo = "    enddo"

        with open(filename, 'w') as f:
            f.write("!!NOTE: This subroutine assumes the all the input arrays are already offloaded to GPU\n")
            self.write_fortran_header(f)

            f.write("!$omp target teams distribute parallel do\n")
            f.write("   do i=1,veclen\n")
            f.write(f"       do L=1,{self.chem.n_species_red}\n")
            f.write("           wdot(i,L) = 0.0d0\n")
            f.write("       enddo\n")
            f.write("   enddo\n")
            f.write("!$omp end target teams distribute parallel do\n")
            
            f.write(self.ytoc_expr)
            f.write("\n")
            
            f.write("    ! Exponential G calculations\n")
            f.write(self.exp_g_expr)
            f.write("\n")

            rnum = 0
            if write_rtypes_together:
                for rtype in self.chem.reaction_types:
                    f.write(f"    ! Reaction type: {rtype}\n")
                    for reaction_number, reaction_expr in self.reaction_expressions.items():
                        if self.chem.reactions[reaction_number]["type"] == rtype:
                            if rnum == 0:
                                f.write(omp_startdo+"\n")
                                f.write(startdo+"\n")
                            elif rnum%10 == 0:
                                f.write(enddo+"\n")
                                f.write(omp_enddo+"\n")
                                f.write(omp_startdo+"\n")
                                f.write(startdo+"\n")
                            f.write(f"    ! Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                            f.write(self.reaction_expressions[reaction_number])
                            f.write("\n")
                            rnum += 1
            else:
                for reaction_number, reaction_expr in self.reaction_expressions.items():
                    if rnum == 0:
                        f.write(omp_startdo+"\n")
                        f.write(startdo+"\n")
                    elif rnum%10 == 0:
                        f.write(enddo+"\n")
                        f.write(omp_enddo+"\n")
                        f.write(omp_startdo+"\n")
                        f.write(startdo+"\n")
                    f.write(f"    ! Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                    f.write(self.reaction_expressions[reaction_number])
                    f.write("\n")
                    rnum += 1

            f.write(enddo+"\n")
            f.write(omp_enddo+"\n")
            f.write("end subroutine getrates_gpu\n")

    def _write_chemgen_mod(self, filename, write_rtypes_together,input_MW):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        template = env.get_template('chemgen_mod.j2')
        
        # Common arrays for all versions
        common_arrays = [
            {"name": "mw_h", "dtype": "real(c_double)"},
            {"name": "smh_coef_h", "dtype": "real(c_double)"},
            {"name": "T_mid_h", "dtype": "real(c_double)"}
        ]

        # Standard reaction arrays
        standard_arrays = [
            {"name": "A_h", "dtype": "real(c_double)"},
            {"name": "B_h", "dtype": "real(c_double)"},
            {"name": "sk_map_h", "dtype": "integer(c_int)"},
            {"name": "sk_coef_h", "dtype": "real(c_double)"},
            {"name": "map_r_h", "dtype": "integer(c_int)"},
            {"name": "coef_r_h", "dtype": "real(c_double)"},
            {"name": "map_p_h", "dtype": "integer(c_int)"},
            {"name": "coef_p_h", "dtype": "real(c_double)"}
        ]

        # Troe reaction arrays
        troe_arrays = [
            {"name": "A_0_troe_h", "dtype": "real(c_double)"},
            {"name": "B_0_troe_h", "dtype": "real(c_double)"},
            {"name": "A_inf_troe_h", "dtype": "real(c_double)"},
            {"name": "B_inf_troe_h", "dtype": "real(c_double)"},
            {"name": "sk_map_troe_h", "dtype": "integer(c_int)"},
            {"name": "sk_coef_troe_h", "dtype": "real(c_double)"},
            {"name": "map_r_troe_h", "dtype": "integer(c_int)"},
            {"name": "coef_r_troe_h", "dtype": "real(c_double)"},
            {"name": "map_p_troe_h", "dtype": "integer(c_int)"},
            {"name": "coef_p_troe_h", "dtype": "real(c_double)"},
            {"name": "eff_fac_troe_h", "dtype": "real(c_double)"},
            {"name": "fcent_coef_troe_h", "dtype": "real(c_double)"}
        ]

        # Third body reaction arrays
        third_body_arrays = [
            {"name": "A_third_h", "dtype": "real(c_double)"},
            {"name": "B_third_h", "dtype": "real(c_double)"},
            {"name": "sk_map_third_h", "dtype": "integer(c_int)"},
            {"name": "sk_coef_third_h", "dtype": "real(c_double)"},
            {"name": "map_r_third_h", "dtype": "integer(c_int)"},
            {"name": "coef_r_third_h", "dtype": "real(c_double)"},
            {"name": "map_p_third_h", "dtype": "integer(c_int)"},
            {"name": "coef_p_third_h", "dtype": "real(c_double)"},
            {"name": "eff_fac_third_h", "dtype": "real(c_double)"}
        ]

        # PLOG reaction arrays
        plog_arrays = [
            {"name": "A_plog_h", "dtype": "real(c_double)"},
            {"name": "B_plog_h", "dtype": "real(c_double)"},
            {"name": "sk_map_plog_h", "dtype": "integer(c_int)"},
            {"name": "sk_coef_plog_h", "dtype": "real(c_double)"},
            {"name": "map_r_plog_h", "dtype": "integer(c_int)"},
            {"name": "coef_r_plog_h", "dtype": "real(c_double)"},
            {"name": "map_p_plog_h", "dtype": "integer(c_int)"},
            {"name": "coef_p_plog_h", "dtype": "real(c_double)"}
        ]

        # Build cp_const_vars list based on which reaction types exist
        cp_const_vars = common_arrays.copy()
        if len(self.chem.get_reactions_by_type("standard")) > 0:
            cp_const_vars.extend(standard_arrays)
        if len(self.chem.get_reactions_by_type("troe")) > 0:
            cp_const_vars.extend(troe_arrays)
        if len(self.chem.get_reactions_by_type("third_body")) > 0:
            cp_const_vars.extend(third_body_arrays)
        if len(self.chem.get_reactions_by_type("plog")) > 0:
            cp_const_vars.extend(plog_arrays)

        # For V4 version: filter out map arrays and add wdot coefficients
        cp_const_vars_v4 = [var for var in cp_const_vars if not "map" in var["name"]]
        
        # Add wdot coefficient arrays for V4
        aname = {"standard":"","troe":"_troe","third_body":"_third","plog":"_plog"}
        for rtype in ["standard", "troe", "third_body", "plog"]:
            if len(self.chem.get_reactions_by_type(rtype)) > 0:
                cp_const_vars_v4.append({"name": f"wdot_coef{aname[rtype]}_h", "dtype": "real(c_double)"})

        context = {
            'n_species_red': self.chem.n_species_red,
            'n_species_sk': self.chem.n_species_sk,
            'cp_const_vars': cp_const_vars,
            'cp_const_vars_v4': cp_const_vars_v4,
            'input_MW': input_MW
        }

        rendered = template.render(**context)
        thread_private = "private(kf,kb,rr,M,k0,kinf,Pr,Fcent,C1,N,F1,F,logPr,logFcent,smh,kfl,kfh,kbl,logPl,logPh,i,L)"
        omp_startdo = f"!$omp target teams distribute parallel do {thread_private}"
        omp_enddo = "!$omp end target teams distribute parallel do"
        startdo = "    do i = 1,veclen"
        enddo = "    enddo"
        with open(filename, 'w') as f:
            f.write(rendered.lstrip())
            f.write("\n")
            f.write("!!NOTE: This subroutine assumes the all the input arrays are already offloaded to GPU\n")
            self.write_fortran_header(f,input_MW=input_MW)

            f.write("!$omp target teams distribute parallel do\n")
            f.write("   do i=1,veclen\n")
            f.write(f"       do L=1,{self.chem.n_species_red}\n")
            f.write("           wdot(i,L) = 0.0d0\n")
            f.write("       enddo\n")
            f.write("   enddo\n")
            f.write("!$omp end target teams distribute parallel do\n")
            
            f.write(self.ytoc_expr)
            f.write("\n")
            
            f.write("    ! Exponential G calculations\n")
            f.write(self.exp_g_expr)
            f.write("\n")

            rnum = 0
            if write_rtypes_together:
                for rtype in self.chem.reaction_types:
                    f.write(f"    ! Reaction type: {rtype}\n")
                    for reaction_number, reaction_expr in self.reaction_expressions.items():
                        if self.chem.reactions[reaction_number]["type"] == rtype:
                            if rnum == 0:
                                f.write(omp_startdo+"\n")
                                f.write(startdo+"\n")
                            elif rnum%10 == 0:
                                f.write(enddo+"\n")
                                f.write(omp_enddo+"\n")
                                f.write(omp_startdo+"\n")
                                f.write(startdo+"\n")
                            f.write(f"    ! Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                            f.write(self.reaction_expressions[reaction_number])
                            f.write("\n")
                            rnum += 1
            else:
                for reaction_number, reaction_expr in self.reaction_expressions.items():
                    if rnum == 0:
                        f.write(omp_startdo+"\n")
                        f.write(startdo+"\n")
                    elif rnum%10 == 0:
                        f.write(enddo+"\n")
                        f.write(omp_enddo+"\n")
                        f.write(omp_startdo+"\n")
                        f.write(startdo+"\n")
                    f.write(f"    ! Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                    f.write(self.reaction_expressions[reaction_number])
                    f.write("\n")
                    rnum += 1

            f.write(enddo+"\n")
            f.write(omp_enddo+"\n")
            f.write("end subroutine getrates_omp_gpu\n")
            f.write("end module\n")
            f.write("#endif\n")
            

    def write_python_header(self, f):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        template = env.get_template('python_header.j2')
        context = {
            'subroutine_name': 'getrates_i' if self.vec else ('getrates_omp_gpu' if self.omp and self.mod else 'getrates_gpu' if self.omp else 'getrates'),
            'vec': self.vec,
            'omp': self.omp,
            'mod': self.mod,
            'n_species_red': self.chem.n_species_red,
            'n_species_sk': self.chem.n_species_sk,
            'Rc': self.Rc,
            'R0': self.R0,
            'Patm': self.Patm
        }

        rendered = template.render(**context)
        f.write(rendered.lstrip())
        f.write("\n")
        
        
    def write_fortran_header(self, f, input_MW=False):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        template = env.get_template('ftn_header.j2')
        context = {
            'subroutine_name': 'getrates_i' if self.vec else ('getrates_omp_gpu' if self.omp and self.mod else 'getrates_gpu' if self.omp else 'getrates'),
            'vec': self.vec,
            'omp': self.omp,
            'mod': self.mod,
            'n_species_red': self.chem.n_species_red,
            'n_species_sk': self.chem.n_species_sk,
            'Rc': self.Rc,
            'R0': self.R0,
            'Patm': self.Patm,
            'input_MW': input_MW
        }

        rendered = template.render(**context)
        f.write(rendered.lstrip())
        f.write("\n")

            
def write_chemistry_mini_app(chem,dirname,ng,ncpu=64,ngpu=8,nt=100):
    if not os.path.exists(dirname):
        os.makedirs(dirname,exist_ok=True)
    filename = os.path.join(dirname,"main.f90")
    with open(filename,'w') as f:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        template = env.get_template('chemistry_mini_app.j2')
        context = {
                "ng":ng,
                "ncpu":ncpu,
                "ngpu":ngpu,
                "nt":nt,
                'n_species_red': chem.n_species_red,
                'n_species_sk': chem.n_species_sk,
        }
        rendered = template.render(**context)
        f.write(rendered.lstrip())
        f.write("\n")
    ##write omp gpu
    chem_expr = chemistry_expressions(chem,omp=True,language="fortran",mod=False)
    filename = os.path.join(dirname,"getrates_gpu.f90")
    chem_expr.write_expressions_to_file(filename,write_rtypes_together=True)

    ##write scalar f90
    chem_expr =chemistry_expressions(chem,omp=False,vec=False,language="fortran")
    filename = os.path.join(dirname,f"getrates.f90")
    chem_expr.write_expressions_to_file(filename,write_rtypes_together=True)
    #write vector f90
    chem_expr =chemistry_expressions(chem,omp=False,vec=True,language="fortran")
    filename = os.path.join(dirname,f"getrates_i.f90")
    chem_expr.write_expressions_to_file(filename,write_rtypes_together=True)


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
