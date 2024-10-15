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
            raise ValueError("vec not implemented yet")

    def create_expressions(self):
        for reaction_number, reaction in self.chem.reactions.items():
            self.reaction_expressions[reaction_number] = self.create_reaction_expression(reaction_number, reaction)
        self.create_ytoc_expr()
        self.create_exp_g_expr()

    def create_exp_g_expr(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        env.globals["abs"] = abs
        template = env.get_template("exp_g.j2")
        
        context = {
            "species_dict": {sp_name:self.chem.species_dict[sp_name].input_data["thermo"] for sp_name in self.chem.species}
        }
        rendered_string = template.render(context)
        self.exp_g_expr = '\n'.join('    ' + line for line in rendered_string.split('\n') if line.strip())

    def create_ytoc_expr(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = Environment(loader=FileSystemLoader(os.path.join(current_dir, "templates")))
        template = env.get_template("ytoc.j2")
        
        context = {
            "chem": {
                "nspecies": self.chem.n_species_red,
                "mw": [self.chem.species_dict[sp].molecular_weight for sp in self.chem.reduced_species],
                "has_third_body_reactions": any(r["type"] == "third_body" for r in self.chem.reactions.values()),
                "has_troe_reactions": any(r["type"] == "troe" for r in self.chem.reactions.values())
            }
        }
        rendered_string = template.render(context)
        self.ytoc_expr = '\n'.join('    ' + line for line in rendered_string.split('\n') if line.strip())

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

        if reaction["type"] == "standard":
            template = env.get_template("reactions/standard.j2")
        elif reaction["type"] == "third_body":
            template = env.get_template("reactions/third_body.j2")
        elif reaction["type"] == "troe":
            template = env.get_template("reactions/troe.j2")
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
            template = env.get_template("reactions/plog.j2")
        
        rendered_string = template.render(context)
        expr = '\n'.join('    ' + line.rstrip() for line in rendered_string.split('\n') if line.strip())
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
            f.write(self.ytoc_expr)
            f.write("\n")
            
            # Write exp_g expressions
            f.write("    # Exponential G calculations\n" if self.language == 'python' else "    ! Exponential G calculations\n")
            f.write(self.exp_g_expr)
            f.write("\n")
            
            for reaction_number, reaction_expr in self.reaction_expressions.items():
                f.write(f"    {'#' if self.language == 'python' else '!'} Reaction {self.chem.reactions[reaction_number]['eqn']}\n")
                f.write(self.reaction_expressions[reaction_number])
                f.write("\n")

            if self.language == 'python':
                f.write("    return kf, kb, rr\n")
            else:  # Fortran
                f.write("end subroutine getrates\n")

        print(f"Expressions and getrates function written to {filename}")


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
            f.write(f"    M = np.zeros(veclen)\n")
            # Add intermediate variables for troe and plog reactions
            f.write("    k0 = np.zeros(veclen)\n")
            f.write("    kinf = np.zeros(veclen)\n")
            f.write("    Pr = np.zeros(veclen)\n")
            f.write("    Fcent = np.zeros(veclen)\n")
            f.write("    C1 = np.zeros(veclen)\n")
            f.write("    N = np.zeros(veclen)\n")
            f.write("    F1 = np.zeros(veclen)\n")
            f.write("    F = np.zeros(veclen)\n")
            f.write("    logPr = np.zeros(veclen)\n")
            f.write("    logFcent = np.zeros(veclen)\n")
            f.write("    smh = np.zeros(veclen)\n")
            f.write("    wdot[:,:] = 0.0\n")
        else:
            f.write("def getrates(T, Y, P, wdot):\n")
            f.write(f"    C = np.zeros({self.chem.n_species_red})\n")
            f.write(f"    EG = np.zeros({self.chem.n_species_sk})\n")
            f.write("    kf = 0.0\n")
            f.write("    kb = 0.0\n")
            f.write("    rr = 0.0\n")
            f.write("    ctot = 0.0\n")
            f.write("    M = 0.0\n")
            # Add intermediate variables for troe and plog reactions
            f.write("    k0 = 0.0\n")
            f.write("    kinf = 0.0\n")
            f.write("    Pr = 0.0\n")
            f.write("    Fcent = 0.0\n")
            f.write("    C1 = 0.0\n")
            f.write("    N = 0.0\n")
            f.write("    F1 = 0.0\n")
            f.write("    F = 0.0\n")
            f.write("    logPr = 0.0\n")
            f.write("    logFcent = 0.0\n")
            f.write("    smh = 0.0\n")
            f.write("    wdot[:] = 0.0\n")
        f.write(f"    Rc = {self.Rc}\n")
        f.write(f"    R0 = {self.R0}\n")
        f.write(f"    Patm = {self.Patm}\n")
        f.write("    kfl = 0.0\n")
        f.write("    kfh = 0.0\n")
        f.write("    kbl = 0.0\n")
        f.write("    logPl = 0.0\n")
        f.write("    logPh = 0.0\n")
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
            f.write("    real(kind=8), dimension(veclen) :: M\n")
            # Add intermediate variables for troe and plog reactions
            f.write("    real(kind=8), dimension(veclen) :: k0, kinf, Pr, Fcent\n")
            f.write("    real(kind=8), dimension(veclen) :: C1, N, F1, F\n")
            f.write("    real(kind=8), dimension(veclen) :: logPr, logFcent\n")
        else:
            f.write("subroutine getrates(P, T, Y, ickwrk, rckwrk, wdot)\n")
            f.write("    implicit none\n")
            f.write("    real(kind=8), intent(in) :: T, P\n")
            f.write(f"    real(kind=8), dimension({self.chem.n_species_red}), intent(in) :: Y\n")
            f.write(f"    real(kind=8), intent(in) :: ickwrk(*),rckwrk(*)\n")
            f.write(f"    real(kind=8), dimension({self.chem.n_species_red}), intent(out) :: wdot\n")
            f.write(f"    real(kind=8), dimension({self.chem.n_species_red}) :: C\n")
            f.write(f"    real(kind=8), dimension({self.chem.n_species_sk}) :: EG\n")
            
            f.write("    real(kind=8) :: kf, kb\n")
            f.write("    real(kind=8) :: rr, ctot, pfac\n")
            f.write("    real(kind=8) :: M, smh\n")
            # Add intermediate variables for troe and plog reactions
            f.write("    real(kind=8) :: k0, kinf, Pr, Fcent\n")
            f.write("    real(kind=8) :: C1, N, F1, F\n")
            f.write("    real(kind=8) :: logPr, logFcent\n")
        f.write(f"    real(kind=8), parameter :: Rc = {self.Rc}D0\n")
        f.write(f"    real(kind=8), parameter :: R0 = {self.R0}D0\n")
        f.write(f"    real(kind=8), parameter :: Patm = {self.Patm}D0\n")
        f.write("    real(kind=8) :: kfl, kfh, kbl, logPl, logPh\n")
        f.write("    integer :: i\n")
        f.write("\n")
        f.write("    pfac = Patm / (R0 * T)\n")
        f.write("    wdot = 0.0d0\n")



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
