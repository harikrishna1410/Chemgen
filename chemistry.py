from parser import ckparser
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

class chemistry:
    def __init__(self,fname,build_graph=False,build_expr=False):

        
        assert isinstance(parser,ckparser)

        self.__reactions_dict = parser.parse_reactions()
        ###this does not give the index of the species. use stoi for that
        ###these are for the whole species list
        self.__species_list = parser.parse_species()
        self.__thermo_data = parser.parse_thermo()
        ##create a specname to idx map
        self.__stoi = {s:i for i,s in enumerate(self.__species_list)}
        ##cretae a idx to a specname map
        self.__itos = {i:s for i,s in enumerate(self.__species_list)}
        ##create chemistry graph
        self.__chemistry_graph = None
        if(build_graph):
            self.create_chemistry_graph()
        ##qssa species
        self.__qssa_species = None
        self.__non_qssa_species = None
        

    @property 
    def reactions(self):
        return self.__reactions_dict
    
    @property
    def species(self):
        return self.__species_list

    @property
    def qssa_species(self):
        return self.__qssa_species

    def species_index(self,species_name):
        return self.__stoi[species_name]

    def species_name(self,species_idx):
        return self.__itos[species_idx]

    """
    function to validate the species maps
    """
    def validate_species_maps(self):
        ##check is itos and stoi are equal
        for s in self.__stoi.keys():
            if(self.__itos[self.__stoi[s]] != s):
                raise ValueError("itos and stoi are not equal")
        ##next check if every index has a species associated with it
        if(len(self.__stoi) != len(self.__species_list) or \
            len(self.__itos) != len(self.__species_list)):
            raise ValueError("itos/stoi don't have all the species.")
        return
    
    """
    function to reset stoi map
    """            
    def reset_stoi_map(self,stoi):
        self.__stoi = stoi
        for s,i in self__stoi.items():
            self.__itos[i] = s
        self.validate_species_maps()
        return

   
    """
    function creates a graph of the whole chemistry
    can be used to mech reduction etc
    """ 
    def create_chemistry_graph(self):
        ##by default I assume a reversible reaction. so only using non-directional graph
        self.__chem_graph = nx.MultiGraph()
        ##add species as nodes
        self.__chem_graph.add_nodes_from(self.__species_list)
        for r in self.__reactions_dict.keys():
            rdict = self.__reactions_dict[r]
            edges = [(i,j) for i in rdict["reacts"] for j in rdict["prods"]]
            keys = [k for k in rdict.keys() if k not in ["reacts","prods"]]
            add_dict = {k:rdict[k] for k in keys}
            self.__chem_graph.add_edges_from([(i,j,add_dict) for i,j in edges])
        return
  
    """
    given the list of the qssa species, this function will change the species to index mapping
    where the first n species are non qssa and the next n_qssa are non-qssa species
    """
    def remap_stoi_for_qssa(self):
        assert self.__qssa_species is not None
        assert self.__non_qssa_species is not None
        qssa_specs = self.__qssa_species
        non_qssa = [s for s in self.__species_list if s not in qssa_specs]
        non_qssa = self.__non_qssa_species
        stoi = {s:i for i,s in enumerate(non_qssa+qssa_specs)}
        self.reset_stoi_map(stoi)
        return

    
    """
    TODO:
    automatically identify the potential species that can be put into qssa
    """ 
    def identify_qssa(self):
        return

    def init_qssa(self,qssa_specs):
        self.__qssa_species = qssa_specs
        self.__non_qssa_species = [s for s in self.__species_list if s not in qssa_specs]
        return
   


"""
chemistry utils
"""

"""
split the whole reactions into reversible/irrev reactions
"""
def filter_reversible_reactions(chem):
    assert isinstance(chem,chemistry)
    rev_dict = {}
    irrev_dict = {}
    for rnum,r_dict in chem.reactions.items():
        if("=>" in r_dict["eqn"] \
            and "<=>" not in r_dict["eqn"]):
            irrev_dict[rnum] = r_dict
        else:
            rev_dict[rnum] = r_dict
    return rev_dict,irrev_dict


"""
returns the maximum number of species in a reaction in the whole mech
""" 
def find_max_specs(chem):
    assert isinstance(chem,chemistry)
    max_specs = 0
    max_reacts = 0
    max_prods = 0
    for rnum,r_dict in chem.reactions.items():
        reacts_dict,prods_dict=r_dict["reacts"],r_dict["prods"]
        max_specs = max(max_specs,len(list(set(list(reacts_dict.keys())+list(prods_dict.keys())))))
        max_reacts = max(max_reacts,len(list(reacts_dict.keys())))
        max_prods = max(max_prods,len(list(prods_dict.keys())))
    return max_specs,max_reacts,max_prods

"""
function to return the maximum number og third body species in a reaction in the whole mech
"""
def find_max_third_body(chem):
    assert isinstance(chem,chemistry)
    max_reacts = 0
    for rnum,r_dict in chem.reactions.items():
        if("third-body" in r_dict.keys()):
            reacts_dict=r_dict["third-body"]
            max_reacts = max(max_reacts,len(list(reacts_dict.keys())))
    return max_reacts

"""
split the whole mech into qssa and non_qssa_reacts
"""
def filter_qssa_reactions(chem):
    assert isinstance(chem,chemistry)
    assert chem.qssa_species is not None
    ##
    qssa_dict = {}
    non_qssa_dict = {}
    for rnum,r_dict in chem.reactions.keys():
        reacts = list(r_dict["reacts"].keys())
        prods = list(r_dict["prods"].keys())
    
        my_qssa = list(set(chem.qssa_specs) & set(reacts+prods))
        if(len(my_qssa)>0):
            qssa_dict[rnum] = r_dict
        else:
            non_qssa_dict[rnum] = r_dict
    return non_qssa_dict,qssa_dict 

"""
split whole mech into std,troe,plog,third
"""
def filter_reactions(chem):
    assert isinstance(chem,chemistry)
    ###filter troe
    std_dict = {}
    troe_dict = {}
    plog_dict = {}
    third_dict = {}
    
    for rnum,r_dict in chem.reactions.items():
        if("troe" in r_dict[rnum].keys()):
            troe_dict[rnum] =  r_dict
        elif("PLOG" in r_dict[rnum].keys()):
            plog_dict[rnum] = r_dict
        elif("third-body" in r_dict[rnum].keys()):
            third_dict[rnum] = r_dict
        else:
            std_dict[rnum] = r_dict

    return std_dict,troe_dict,plog_dict,third_dict
#"""
#    class to hold the reaction expressions.
#"""
#
#class reaction:
#    def __init__(self,type):
#
#        self.__type = type
#
#        ###build expressions
#        self.rf_expr = None
#        self.rb_expr = None
#
#        self.__build_std_expressions()
#
#    """
#        this function builds the reaction rate expressions using 
#        sympy based on the reaction type
#        Advantages:
#        1. very flexible to try out new expressions for the same reaction rates
#           or try out a an approximate expressions. Maybe using a linear expression
#           for some reactions is good enough.
#
#        Return:
#        returns expressions for forward and backward reaction rate.
#        Note these are the actual ones
#    """
#    
#    def __build_std_expressions(self):
#        if(self._type =="arh"):
#            self.__build_std_arh_expr()
#        elif(self.__type == "plog"):
#            self.__build_std_plog_expr()
#        elif(self.__type == "troe"):
#            self.__build_std_troe_expr()
#        elif(self.__type == "third-body"):
#            self.__buid_std_third_expr()
#        else:
#            raise ValueError("unknown reaction type {self.__type}")
#    
#    """
#        function to build a std arh expression
#    """
#
#    def __build_std_arh_epr(self):
#        self.rf_expr = None
#        self.rb_expr = None 
#    
#    """
#        function to build a std plog expression
#    """
#
#    def __build_std_plog_epr(self):
#        self.rf_expr = None
#        self.rb_expr = None 
#    
#    """
#        function to build a std troe expression
#    """
#
#    def __build_std_troe_epr(self):
#        self.rf_expr = None
#        self.rb_expr = None 
#    
#    """
#        function to build a std third expression
#    """
#
#    def __build_std_third_epr(self):
#        self.rf_expr = None
#        self.rb_expr = None 
#
#    
#    """
#        approximate standard expressions
#        options: 
#            taylor: approximate using taylor series expression 
#            sub: substitute using some constant values
#    """
#
#    def approx_std_expressions(self,params,method="taylor"):
#        
#        if(method == "taylor"):
#            self.__taylor_approx_expr(params)
#        elif(method == "sub"):
#            self.rf_expr = None
#            self.rb_expr = None
#        else:
#            raise ValueError("unknown method {method}")
#
#    """
#        approximate standard chemistry expressions using
#        taylors series approximation
#    """
#
#    def __taylor_approx_expr(self,params):
#        self.rf_expr = None
#        self.rb_expr = None
