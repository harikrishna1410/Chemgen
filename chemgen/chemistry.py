from .parser import ckparser
from .utils import calculate_nsp_per_block, calculate_nreact_per_block
import cantera as ct
import networkx as nx
import sympy as sp

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
REACTION_TYPES = ["standard", "troe", "third_body", "plog"]
class chemistry:
    def __init__(self, ck_file, therm_file, parser, build_graph=False,qssa_species=[]):
        
        assert isinstance(parser, ckparser)

        self.__parser = parser
        self.__reactions_dict = parser.parse_reactions(ck_file, therm_file)
        self.__species_list = parser.parse_species(ck_file)
        self.__qssa_species = qssa_species
        self.__reduced_species_list = [i for i in self.__species_list if i not in self.__qssa_species]
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
    
    def find_max_specs(self):
        max_specs = 0
        max_reacts = 0
        max_prods = 0
        for r_dict in self.reactions.values():
            reacts_dict, prods_dict = r_dict["reacts"], r_dict["prods"]
            max_specs = max(max_specs, len(set(list(reacts_dict.keys()) + list(prods_dict.keys()))))
            max_reacts = max(max_reacts, len(reacts_dict.keys()))
            max_prods = max(max_prods, len(prods_dict.keys()))
        return max_specs, max_reacts, max_prods

    def find_max_third_body(self):
        max_reacts = 0
        for r_dict in self.get_reactions_by_type("third_body").values():
            if "third_body" in r_dict:
                reacts_dict = r_dict["third_body"]
                max_reacts = max(max_reacts, len(reacts_dict.keys()))
        return max_reacts
    

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
