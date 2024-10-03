import re
import sys
import numpy as np
import torch
import sympy as sp

"""
    chemistry neural network

    layer 1: compute forward rate
    layer 2: compute backward rate
    layer 3: compute wdot
"""

class chemistry_nn(torch.nn.Module):
    def __init__(self,
                 nreacts,
                 nspecies,
                 type,
                 logA,
                 Ea_R,
                 beta,
                 react_ij,
                 prod_ij):
        
        super().__init__()

        ###init first layer
        self.logkf_layer = torch.nn.Linear(2,nreacts)
        self.logkf_layer.weight = torch.column_stack((Ea_R,beta))
        self.logkf_layer.bias = logA
        ###
        self.logkb_layer = torch.nn.Linear(nspecies+nreacts,nreacts)
        w1 = -react_ij + prod_ij ###nreacts x nspecies
        w2 = torch.eye(nreacts)
        w1 = torch.column_stack((w1,w2)) ###nreacts x (nspecies + nreacts)
        self.logkb_layer.weight = w1
        self.logkb_layer.bias = torch.zeros(nreacts)

        ###
        self.logrf_layer = torch.nn.Linear(nspecies+nreacts,nreacts)
        w1 = torch.column_stack((react_ij,np.eye(nreacts)))
        self.logrf_layer.weight = w1
        self.logrf_layer.bias = torch.zeros(nreacts)
        ###
        self.logrb_layer = torch.nn.Linear(nspecies+nreacts,nreacts)
        w1 = torch.column_stack((prod_ij,np.eye(nreacts)))
        self.logrb_layer.weight = w1
        self.logrb_layer.bias = torch.zeros(nreacts)
        ###
        self.wdot_layer = torch.nn.Linear(nreacts,nspecies)
        w1 = -torch.transpose(react_ij) + torch.transpose(prod_ij) ###nspecies x nreacts
        self.wdot_layer.weight = w1
        self.wdot_layer.bias = torch.zeros(nspecies)
        ###
        if(type == "arh"):
            self.forward = self.__arh_forward_pass
        else:
            raise ValueError("Unsupported reaction type {type}")
        return
    
    """
        This the function that computes the reaction rate
        for the arh equations

        ###Eq_R_nu_ij are the weights of first hidden layer
        ###logA is the bias of the first hidden layer
        ##C_T is the input for both first and second layers
    """
    def __arh_forward(self,
                      T,
                      C,
                      logEG):
        
        logT = torch.log(T)
        T_inv = -torch.reciprocal(T)
        logkf = self.logkf_layer(torch.column_stack((T_inv,logT)))
        ### logEG.shape = ng x nspecies
        logkb = self.logkb_layer(torch.column_stack(logEG,logkf))
        ##
        rr = torch.exp(logkf) - torch.exp(logkb)
        ##
        wdot = self.wdot_layer(rr)

        return wdot
