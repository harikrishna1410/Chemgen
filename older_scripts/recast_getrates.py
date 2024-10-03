###this script reads the chemkin files and list of qssa species to recast the 
###getrates into a matrix format propsed in the below paper
###Barwey, S., Raman, V., 2021. A Neural Network-Inspired Matrix Formulation of Chemical Kinetics for Acceleration on GPUs. Energies 14, 2710. https://doi.org/10.3390/en14092710

from add_species_and_reactions import *


extra_space = "         "
space = "      "
space_m1 = "     "

###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def add_trailing_and(line,a=False):
    line = line + " "*(71-len(line))
    if(a):
        line = line + "&\n"
    else:
        line = line + "\n"
    return line
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
###writes the function to calculate the forward reactions
def write_forward_rate_const_func(inp_file,out_file):
    print("writing forward reaction rate")
    ####find all species and reactions 
    spec = find_sk_species(inp_file)
    ###order_change = True puts all standard,falloff,third-body,plog reactions
    r_dict,n_st,n_troe,n_third,n_plog \
            = parse_reactions(inp_file,change_order=True)
    nspec = len(spec)
    nreacts = len(r_dict.keys())
    print("num reacts ",nreacts)
    print(r_dict.keys())
    ### check if all reactions are valid
    print("testing reaction valididty")
    for rnum in r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        for s in reacts+prods:
            if(s not in spec):
                print("ERROR: %s is not a valid species"%(s))
                sys.exit()
    ##first put all the lines declerations and stuff
    new_lines = []
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    new_lines.append("%sSUBROUTINE GET_FORWARD_RATE_CONST(Np,Nr,TEMP,logKF)\n"%(space))
    new_lines.append("%sIMPLICIT NONE\n"%(space))
    new_lines.append("%sINTEGER,INTENT(IN) :: Np,Nr\n"%(space))
    new_lines.append("%sREAL,DIMENSION(Np),INTENT(IN) :: TEMP\n"%(space))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(OUT) :: logKF\n"%(space,nreacts))
    new_lines.append("!%s*****local declerations******\n"%(space_m1))
    new_lines.append("%sINTEGER :: rnum,idx \n"%(space))
    new_lines.append("%sREAL,DIMENSION(%d) :: A_1D=[&\n"%(space,nreacts-n_plog))
    #new_lines.append("%sREAL,DIMENSION(2,Nr) :: B_2D\n"%(space))
    ####things starts happening here
    ##
    ##
    ###create frequency factor array for all the reactions
    A_lines = []
    #line = "%sA_1D = ("%(space)
    #line = add_trailing_and(line)
    #A_lines.append(line)
    ###
    B_lines = []
    #line = "%sB_1D = ("%(space)
    #line = add_trailing_and(line)
    #B_lines.append(line)
    ###ADD NON PLOG stuff
    for rnum in range(1,len(r_dict.keys())+1):
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        if("troe" in r_dict[rnum].keys() or "PLOG" in r_dict[rnum].keys()\
            or "third-body" in r_dict[rnum].keys()):
            print("ERROR: %d %s is pressure dependent"%(rnum,r_dict[rnum]["eqn"]))
            print("reaction dict keys ",r_dict[rnum].keys())
            if("PLOG" in r_dict[rnum].keys()):
                continue
            ##sys.exit()
        ###add RF
        ##A
        A = r_dict[rnum]["arh"][0]
        beta = r_dict[rnum]["arh"][1]
        Ea = r_dict[rnum]["arh"][2]
        ###beta
        if(beta>0):
            sign = "+"
        else:
            sign = "-"
        bline = ("%s%s%21.15E,"%(space,sign,np.abs(beta)))\
                  .replace("E","D")
        ##
        if(rnum != nreacts):
            ###log(A)
            aline = ("%s%21.15E,&"%(space,np.log(A)))\
                      .replace("E","D")
            aline = add_trailing_and(aline)
            ####Ea/R
            if(Ea>0):
                sign="-"
            else:
                sign="+"
            bline = bline + ("%s%21.15E,&"%(sign,np.abs(Ea)/R_c))\
                      .replace("E","D")
            ###
            bline = add_trailing_and(bline)
            
        else:
            aline = ("%s%21.15E]"%(space,np.log(A)))\
                      .replace("E","D")
            aline = add_trailing_and(aline,a=False)
            ####Ea/R
            if(Ea>0):
                sign="-"
            else:
                sign="+"
            bline = bline + ("%s%21.15E]\n"%(sign,np.abs(Ea)/R_c))\
                      .replace("E","D")
            bline = add_trailing_and(bline,a=False)
        A_lines.append(aline)
        B_lines.append(bline)
    ### 
    new_lines = new_lines + A_lines
    new_lines.append("%sREAL,DIMENSION(%d) :: B_1D=[&\n"%(space,2*nreacts))
    new_lines = new_lines + B_lines
    #line = "%sB_2D = RESHAPE(B_1D,(/2,Nr/))\n"
    #lines = lines + line
    ###now mutiply with temperature 
#    line = "!%sfor now I am simply multiplying but in the paper they used cuBLAS\n!%sBut, I am really reluctant to allocate another temperature array of size (npts*2)\n!%sI will see how this goes and decide\n"\
#            %(space_m1,space_m1,space_m1)
#    new_lines.append(line)
    line = "!%scompute all non plog ones which don't have any IF conditions\n"%(space_m1)
    new_lines.append(line)
    #
    line = "%sDo idx=1,Np\n"%(space)
    new_lines.append(line)
#    line = "!%sI did this to have data locality between threads in logKF array as it can be quite big\n"\
#            %(space_m1)
#    new_lines.append(line)
    line = "%sDo rnum=1,%d\n"%(nreacts-n_plog)
    new_lines.append(line)
    ###
    line = "%slogKF(idx,rnum) = log(Temp(idx))*B_1D(2*rnum-1)&"%(space)
    line = add_trailing_and(line)
    new_lines.append(line)
    ##
    line = "%s+ B_1D(2*rnum)/Temp(idx) + A_1D(rnum)\n"%(space)
    new_lines.append(line)
    ##
    line = "%sENDDO\n"%(space)
    new_lines.append(line)
    line = "%sENDDO\n"%(space)
    new_lines.append(line)
    ##
    line = "!%scompute all plog ones\n"%(space_m1)
    new_lines.append(line)
    line = "!%ssince there aren't a lot of plog reactions. Just unroll them\n"%(space_m1)
    new_lines.append(line)
    ####atm -> dyne/cm2
    P_conv = 1013250.0 
    for rnum in range(nreacts-n_plog+1,nreacts+1):
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
            ##sys.exit()
        ###add RF
        ##A
        A = r_dict[rnum]["arh"][0]
        beta = r_dict[rnum]["arh"][1]
        Ea = r_dict[rnum]["arh"][2]
        ###beta
        if(beta>0):
            sign = "+"
        else:
            sign = "-"
        bline = ("%s%s%21.15E,"%(space,sign,np.abs(beta)))\
                  .replace("E","D")
        ##
        if(rnum != nreacts):
            ###log(A)
            aline = ("%s%21.15E,&"%(space,np.log(A)))\
                      .replace("E","D")
            aline = add_trailing_and(aline)
            ####Ea/R
            if(Ea>0):
                sign="-"
            else:
                sign="+"
            bline = bline + ("%s%21.15E,&"%(sign,np.abs(Ea)/R_c))\
                      .replace("E","D")
            ###
            bline = add_trailing_and(bline)
            
        else:
            aline = ("%s%21.15E]"%(space,np.log(A)))\
                      .replace("E","D")
            aline = add_trailing_and(aline,a=False)
            ####Ea/R
            if(Ea>0):
                sign="-"
            else:
                sign="+"
            bline = bline + ("%s%21.15E]\n"%(sign,np.abs(Ea)/R_c))\
                      .replace("E","D")
            bline = add_trailing_and(bline,a=False)
        A_lines.append(aline)
        B_lines.append(bline)
        
    line = "%sEND SUBROUTINE\n"%(space)
    new_lines.append(line)
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    ##write everything
    f=open(out_file,"a")
    for l in new_lines:
        f.write(l)
    f.close()
    return
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def write_backward_rate_const_func(inp_file,therm_file,out_file):
    ###find the skeletal species
    specs = find_sk_species(inp_file)
    nspec = len(specs)
    r_dict = parse_reactions(inp_file,change_order=True)
    nreacts = len(r_dict.keys())
    ##
    new_lines = []
    #new_lines.append("!********************************************************************\n")
    #new_lines.append("!********************************************************************\n")
    #new_lines.append("!********************************************************************\n")
    new_lines.append("%sSUBROUTINE GET_BACKWARD_RATE_CONST(Np,Nr,Ns,TEMP,logKF,logKR)\n"%(space))
    new_lines.append("%sIMPLICIT NONE\n"%(space))
    new_lines.append("%sREAL,PARAMETER :: RU=8.31451D7, PATM=1.01325D6\n"%(space))
    new_lines.append("%sINTEGER,INTENT(IN) :: Np,Nr,Ns\n"%(space))
    new_lines.append("%sREAL,DIMENSION(Np),INTENT(IN) :: TEMP\n"%(space))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(IN) :: logKF\n"%(space,nreacts))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(OUT) :: logKR\n"%(space,nreacts))
    new_lines.append("!%s*******local declerations****************\n"%(space_m1))
    new_lines.append("%sINTEGER :: rnum,idx,L,COEF_SUM \n"%(space))
    new_lines.append("%sINTEGER,DIMENSION(Np,%d) :: HL_MAP\n"%(space,nspec))
    new_lines.append("%sREAL,DIMENSION(Np,%d) :: LOG_EG\n"%(space,nspec))
    new_lines.append("%sREAL :: LOG_EQK\n"%(space))
    #new_lines.append("%sREAL,DIMENSION(%d) :: COEF_1D\n"%(space,14*nspec))
    new_lines.append("%sREAL,DIMENSION(7,%d,2) :: COEF_3D=RESHAPE((/&\n"%(space,nspec))
    ###order in which coefs are actually written
    #T = ["","*TI","*TN(1)","*TN(2)","*TN(3)","*TN(4)","*TN(5)"]
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    write_order = [6,5,0,1,2,3,4]
    prod = [1.0,-1.0,1.0,1/2,1/6,1/12,1/20]
#    new_lines.append("%sCOEF_1D =(\n"%(space))
    for idx,cid in enumerate(write_order):
        for sid,spec_name in enumerate(specs):
            ###get the coefficients
            temp_limits,coef=get_nasa_poly(spec_name,therm_file)
            high_coef = coef[0:7] 
            low_coef = coef[7:14]
            if(idx == 6 and sid == nspec-1):
                line = ("%s%21.15E,%21.15E&\n"%(space,low_coef[cid]*prod[idx]\
                                    ,high_coef[cid]*prod[idx])).replace("E","D")
            else:
                line = ("%s%21.15E,%21.15E,&\n"%(space,low_coef[cid]*prod[idx]\
                                    ,high_coef[cid]*prod[idx])).replace("E","D")
            new_lines.append(line)
    line = "%s/),(/7,%d,2/))\n"%(space,nspec)
    new_lines.append(line)
    #new_lines.append("%sCOEF_3D = RESHAPE(COEF_1D,(/7,%d,2/))\n"%(space,nspec))
    ###
    new_lines.append("%sREAL,DIMENSION(%d) :: TLIM=[&\n"%(space,nspec))
    ####create a map
    #new_lines.append("%sTLIM=(\n"%(space))
    for sid,spec_name in enumerate(specs):
        ###get the coefficients
        temp_limits,coef=get_nasa_poly(spec_name,therm_file)
        if(sid == nspec-1):
            new_lines.append("%s%.2f&\n"%(space,temp_limits[-1]))
        else:
            new_lines.append("%s%.2f,&\n"%(space,temp_limits[-1]))
    new_lines.append("%s]\n"%(space)) 
    new_lines.append("!\n")
    new_lines.append("!%sinstead of doing a lot of multiplications I am just doing a hash map.\n"%(space))
    new_lines.append("!%sthis is the array containing the species id contributing to a reaction.\n"%(space))
    new_lines.append("!%sNOTE: the second dimension is the maximum number of species/per reaction.\n"%(space))
    ##find max specs
    max_specs = 0
    for rnum in range(1,len(r_dict.keys())+1):
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        max_specs = max(max_specs,len(list(set(list(reacts_dict.keys())+list(prods_dict.keys())))))
    
    ###ST_MAP
    ###ST_COEF
    ###
    map_lines = []
    coef_lines = []
    map_line="%s"%(space)
    coef_line="%s"%(space)
    for rnum in range(1,len(r_dict.keys())+1):
        coef_sum = 0
        count = 0
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        ###add reactants side
        for spec_name in reacts_dict.keys():
            spec_idx = specs.index(spec_name)+1
            count = count + 1
            mstr = "%d,"%(spec_idx)
            if(len(map_line)+len(mstr) >= 72):
                map_lines.append(map_line+"&\n")
                map_line =  "%s%s"%(space,mstr)
            else:
                map_line = map_line + mstr
            ###coef lines
            if(spec_name in prods_dict.keys()):
                cstr = "%d,"%(prods_dict[spec_name]-reacts_dict[spec_name])
                coef_sum = coef_sum + prods_dict[spec_name]-reacts_dict[spec_name]
                del prods_dict[spec_name]
            else:
                cstr = "%d,"%(-reacts_dict[spec_name])
                coef_sum = coef_sum - reacts_dict[spec_name]
            if(len(coef_line)+len(cstr) >= 72):
                coef_lines.append(coef_line+"&\n")
                coef_line =  "%s%s"%(space,cstr)
            else:
                coef_line = coef_line + cstr
        ##add products
        for spec_name in prods_dict.keys():
            spec_idx = specs.index(spec_name)+1
            count = count + 1
            mstr = "%d,"%(spec_idx)
            if(len(map_line)+len(mstr) >= 72):
                map_lines.append(map_line+"&\n")
                map_line =  "%s%s"%(space,mstr)
            else:
                map_line = map_line + mstr
            ###coef lines
            cstr = "%d,"%(prods_dict[spec_name])
            coef_sum = coef_sum + prods_dict[spec_name]
            if(len(coef_line)+len(cstr) >= 72):
                coef_lines.append(coef_line+"&\n")
                coef_line =  "%s%s"%(space,cstr)
            else:
                coef_line = coef_line + cstr
        ###add the max - count 
        for idx in range(max_specs-count):
            spec_idx = nspec
            mstr = "%d,"%(spec_idx)
            if(len(map_line)+len(mstr) >= 72):
                map_lines.append(map_line+"&\n")
                map_line =  "%s%s"%(space,mstr)
            else:
                map_line = map_line + mstr
            ###coef lines
            cstr = "0,"
            if(len(coef_line)+len(cstr) >= 72):
                coef_lines.append(coef_line+"&\n")
                coef_line =  "%s%s"%(space,cstr)
            else:
                coef_line = coef_line + cstr
        ###add the sum
        cstr = "%d,"%(coef_sum)
        if(len(coef_line)+len(cstr) >= 72):
            coef_lines.append(coef_line+"&\n")
            coef_line =  "%s%s"%(space,cstr)
        else:
            coef_line = coef_line + cstr
    ##
    coef_lines.append(coef_line) 
    map_lines.append(map_line) 
    if(coef_lines[-1][-1]==","):
        line = coef_lines[-1]
        line = line[:-1]+"&\n"
        coef_lines[-1] =  line
    if(map_lines[-1][-1]==","):
        line = map_lines[-1]
        line = line[:-1]+"&\n"
        map_lines[-1] =  line
    ##
    new_lines.append("%sINTEGER,DIMENSION(%d,%d) :: ST_MAP=RESHAPE((/&\n"%(space,nreacts,max_specs))
    new_lines = new_lines + map_lines
    line = "%s/),(/%d,%d/))\n"%(space,nreacts,max_specs)
    new_lines.append(line)
    ###
    new_lines.append("%sINTEGER,DIMENSION(%d,%d) :: ST_COEF=RESHAPE((/&\n"%(space,nreacts,max_specs+1))
    new_lines = new_lines + coef_lines
    line = "%s/),(/%d,%d/))\n"%(space,nreacts,max_specs+1)
    new_lines.append(line)
    ###
    new_lines.append("!%sI made this map to make the SMH loops SIMD.\n"%(space_m1))
    new_lines.append("!%sThe coef array also has low and high in consecutive memory locations\n"%(space_m1))
    new_lines.append("!\n") 
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO L=1,Ns\n"%(space))
    new_lines.append("%sHL_MAP(idx,L) = INT(TEMP(idx)>TLIM(L))+1\n"%(space))
    new_lines.append("%sENDDO\n"%(space))
    new_lines.append("%sENDDO\n"%(space))
    new_lines.append("!\n")
    ##
    new_lines.append("!%sI am trying to reduce the register/thread so seperate loops.\n"%(space_m1))
    new_lines.append("!\n")
    ## 1
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    new_lines.append("!%s1\n"%(space_m1)) 
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO L=1,Ns\n"%(space))
    new_lines.append("%sLOG_EG(idx,L) = COEF_3D(1,L,HL_MAP(idx,L))\n"%(space)) 
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    ## 2
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    new_lines.append("!%s2\n"%(space_m1)) 
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO L=1,Ns\n"%(space))
    new_lines.append("%sLOG_EG(idx,L) = LOG_EG(idx,L)+COEF_3D(2,L,HL_MAP(idx,L))/TEMP(idx)\n"%(space)) 
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    ## 3
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    new_lines.append("!%s3\n"%(space_m1)) 
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO L=1,Ns\n"%(space))
    new_lines.append("%sLOG_EG(idx,L) = LOG_EG(idx,L)+COEF_3D(3,L,HL_MAP(idx,L))*LOG(TEMP(idx))\n"%(space)) 
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    ## 4
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    new_lines.append("!%s4\n"%(space_m1)) 
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO L=1,Ns\n"%(space))
    new_lines.append("%sLOG_EG(idx,L) = LOG_EG(idx,L)+COEF_3D(4,L,HL_MAP(idx,L))*TEMP(idx)\n"%(space)) 
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    ## 5
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    new_lines.append("!%s5\n"%(space_m1)) 
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO L=1,Ns\n"%(space))
    new_lines.append("%sLOG_EG(idx,L) = LOG_EG(idx,L)+COEF_3D(5,L,HL_MAP(idx,L))*TEMP(idx)**2\n"%(space)) 
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    ## 6
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    new_lines.append("!%s6\n"%(space_m1)) 
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO L=1,Ns\n"%(space))
    new_lines.append("%sLOG_EG(idx,L) = LOG_EG(idx,L)+COEF_3D(6,L,HL_MAP(idx,L))*TEMP(idx)**3\n"%(space)) 
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    ## 7
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    new_lines.append("!%s7\n"%(space_m1)) 
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO L=1,Ns\n"%(space))
    new_lines.append("%sLOG_EG(idx,L) = LOG_EG(idx,L)+COEF_3D(7,L,HL_MAP(idx,L))*TEMP(idx)**4\n"%(space)) 
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("!\n")
    ##
    new_lines.append("!%sNOTE: care should be taken while distributing the inner most loop!!\n"%(space))
    new_lines.append("!%s maybe only distribute the two outermost loops\n"%(space))
    new_lines.append("%sDO idx=1,Np\n"%(space))
    new_lines.append("%sDO rnum = 1,%d\n"%(space,nreacts))
    #new_lines.append("!%slot of additional additions here!!\n"%(space))
    new_lines.append("%sLOG_EQK = 0.0D0\n"%(space))
#    new_lines.append("%sCOEF_SUM = 0\n"%(space))
    new_lines.append("%sDO L = 1,%d\n"%(space,max_specs))
    new_lines.append("%sLOG_EQK = LOG_EQK + ST_COEF(rnum,L)*LOG_EG(idx,ST_MAP(rnum,L))\n"%(space))
#    new_lines.append("%sCOEF_SUM = COEF_SUM + ST_COEF(rnum,L)\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%slogKR(idx,rnum) = logKF(idx,rnum) - &\n"%(space))
    new_lines.append("%sLOG_EQK + ST_COEF(rnum,6)*LOG(PATM/(RU*TEMP(idx)))\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    ##
    new_lines.append("%sEND SUBROUTINE\n"%(space))
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    
    f=open(out_file,"a")
    for l in new_lines:
        f.write(l)
    f.close()

    return
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def write_net_reaction_rate(inp_file,out_file):
    ###find the skeletal species
    specs = find_sk_species(inp_file)
    nspec = len(specs)
    r_dict = parse_reactions(inp_file,change_order=True)
    nreacts = len(r_dict.keys())
    ##
    new_lines = []
    new_lines.append("%sSUBROUTINE GET_NET_REACTION_RATE(Np,logKF,logKR,C,RR)\n"%(space))
    new_lines.append("%sIMPLICIT NONE\n"%(space))
    new_lines.append("%sREAL,PARAMETER :: RU=8.31451D7, PATM=1.01325D6\n"%(space))
    new_lines.append("%sINTEGER,INTENT(IN) :: Np\n"%(space))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(IN) :: logKF\n"%(space,nreacts))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(IN) :: logKR\n"%(space,nreacts))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(IN) :: C\n"%(space,nspec))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(OUT) :: RR\n"%(space,nreacts))
    new_lines.append("!%s*******local declerations****************\n"%(space_m1))
    new_lines.append("%sINTEGER :: rnum,idx,L \n"%(space))
    new_lines.append("%sREAL :: RF,RB \n"%(space))
    ##
    max_reacts = 0
    max_prods = 0
    for rnum in range(1,len(r_dict.keys())+1):
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        max_reacts = max(max_reacts,len(list(reacts_dict.keys())))
        max_prods = max(max_prods,len(list(prods_dict.keys())))
    
    ###ST_MAP
    ###ST_COEF
    ###
    map_lines = []
    coef_lines = []
    map_line="%s"%(space)
    coef_line="%s"%(space)
    for rnum in range(1,len(r_dict.keys())+1):
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        ###add reactants side
        for spec_name in reacts_dict.keys():
            spec_idx = specs.index(spec_name)+1
            mstr = "%d,"%(spec_idx)
            if(len(map_line)+len(mstr) >= 72):
                map_lines.append(map_line+"&\n")
                map_line =  "%s%s"%(space,mstr)
            else:
                map_line = map_line + mstr
            ###coef lines
            cstr = "%d,"%(reacts_dict[spec_name])
            if(len(coef_line)+len(cstr) >= 72):
                coef_lines.append(coef_line+"&\n")
                coef_line =  "%s%s"%(space,cstr)
            else:
                coef_line = coef_line + cstr

        ###add the max - count 
        for idx in range(max_reacts-len(reacts_dict.keys())):
            spec_idx = nspec
            mstr = "%d,"%(spec_idx)
            if(len(map_line)+len(mstr) >= 72):
                map_lines.append(map_line+"&\n")
                map_line =  "%s%s"%(space,mstr)
            else:
                map_line = map_line + mstr
            ###coef lines
            cstr = "0,"
            if(len(coef_line)+len(cstr) >= 72):
                coef_lines.append(coef_line+"&\n")
                coef_line =  "%s%s"%(space,cstr)
            else:
                coef_line = coef_line + cstr
        
    coef_lines.append(coef_line) 
    map_lines.append(map_line) 

    if(coef_lines[-1][-1]==","):
        line = coef_lines[-1]
        line = line[:-1]+"&\n"
        coef_lines[-1] =  line
    if(map_lines[-1][-1]==","):
        line = map_lines[-1]
        line = line[:-1]+"&\n"
        map_lines[-1] =  line
    ##
    new_lines.append("%sINTEGER,DIMENSION(%d,%d) :: MAP_R=RESHAPE((/&\n"%(space,nreacts,max_reacts))
    new_lines = new_lines + map_lines
    line = "%s/),(/%d,%d/))\n"%(space,nreacts,max_reacts)
    new_lines.append(line)
    ###
    new_lines.append("%sINTEGER,DIMENSION(%d,%d) :: COEF_R=RESHAPE((/&\n"%(space,nreacts,max_reacts))
    new_lines = new_lines + coef_lines
    line = "%s/),(/%d,%d/))\n"%(space,nreacts,max_reacts)
    new_lines.append(line)
    ##
    map_lines = []
    coef_lines = []
    map_line="%s"%(space)
    coef_line="%s"%(space)
    for rnum in range(1,len(r_dict.keys())+1):
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        for spec_name in prods_dict.keys():
            spec_idx = specs.index(spec_name)+1
            mstr = "%d,"%(spec_idx)
            if(len(map_line)+len(mstr) >= 72):
                map_lines.append(map_line+"&\n")
                map_line =  "%s%s"%(space,mstr)
            else:
                map_line = map_line + mstr
            ###coef lines
            cstr = "%d,"%(prods_dict[spec_name])
            if(len(coef_line)+len(cstr) >= 72):
                coef_lines.append(coef_line+"&\n")
                coef_line =  "%s%s"%(space,cstr)
            else:
                coef_line = coef_line + cstr
        ###add the max - count 
        for idx in range(max_prods-len(prods_dict.keys())):
            spec_idx = nspec
            mstr = "%d,"%(spec_idx)
            if(len(map_line)+len(mstr) >= 72):
                map_lines.append(map_line+"&\n")
                map_line =  "%s%s"%(space,mstr)
            else:
                map_line = map_line + mstr
            ###coef lines
            cstr = "0,"
            if(len(coef_line)+len(cstr) >= 72):
                coef_lines.append(coef_line+"&\n")
                coef_line =  "%s%s"%(space,cstr)
            else:
                coef_line = coef_line + cstr
    
    coef_lines.append(coef_line) 
    map_lines.append(map_line) 
    ##
    if(coef_lines[-1][-1]==","):
        line = coef_lines[-1]
        line = line[:-1]+"&\n"
        coef_lines[-1] =  line
    if(map_lines[-1][-1]==","):
        line = map_lines[-1]
        line = line[:-1]+"&\n"
        map_lines[-1] =  line
    ##
    new_lines.append("%sINTEGER,DIMENSION(%d,%d) :: MAP_P=RESHAPE((/&\n"%(space,nreacts,max_prods))
    new_lines = new_lines + map_lines
    line = "%s/),(/%d,%d/))\n"%(space,nreacts,max_prods)
    new_lines.append(line)
    ###
    new_lines.append("%sINTEGER,DIMENSION(%d,%d) :: COEF_P=RESHAPE((/&\n"%(space,nreacts,max_prods))
    new_lines = new_lines + coef_lines
    line = "%s/),(/%d,%d/))\n"%(space,nreacts,max_prods)
    new_lines.append(line)
    ##
    new_lines.append("!\n")
    ##
    new_lines.append("%sDO idx = 1,Np\n"%(space))
    new_lines.append("%sDO rnum = 1,%d\n"%(space,nreacts))
    #
    new_lines.append("%sRF=0\n"%(space))
    new_lines.append("%sDO L = 1,%d\n"%(space,max_reacts))
    new_lines.append("%sRF = RF + COEF_R(rnum,L)*LOG(C(idx,MAP_R(rnum,L)))\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sRF = RF + logKF(idx,rnum)\n"%(space))
    #
    new_lines.append("%sRB=0\n"%(space))
    new_lines.append("%sDO L = 1,%d\n"%(space,max_prods))
    new_lines.append("%sRB = RB + COEF_P(rnum,L)*LOG(C(idx,MAP_P(rnum,L)))\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sRB = RB + logKR(idx,rnum)\n"%(space))
    ##
    new_lines.append("%sRR(idx,rnum) = EXP(RF) - EXP(RB)\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND SUBROUTINE\n"%(space))
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    
    f=open(out_file,"a")
    for l in new_lines:
        f.write(l)
    f.close()
    return

###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def write_net_production_rate(inp_file,out_file):
    ###find the skeletal species
    specs = find_sk_species(inp_file)
    nspec = len(specs)
    r_dict = parse_reactions(inp_file,change_order=True)
    nreacts = len(r_dict.keys())
    ##
    new_lines = []
    new_lines.append("%sSUBROUTINE GET_NET_PRODUCTION_RATE(Np,RR,WDOT)\n"%(space))
    new_lines.append("%sIMPLICIT NONE\n"%(space))
    new_lines.append("%sREAL,PARAMETER :: RU=8.31451D7, PATM=1.01325D6\n"%(space))
    new_lines.append("%sINTEGER,INTENT(IN) :: Np\n"%(space))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(IN) :: RR\n"%(space,nreacts))
    new_lines.append("%sREAL,DIMENSION(Np,%d),INTENT(OUT) :: WDOT\n"%(space,nspec))
    new_lines.append("!%s*******local declerations****************\n"%(space_m1))
    new_lines.append("%sINTEGER :: rnum,idx,L \n"%(space))
    ##
    hash_map = np.zeros((nreacts,nspec),dtype=np.int32)
    for rnum in range(1,len(r_dict.keys())+1):
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        for spec_name in reacts_dict.keys():
            spec_idx = specs.index(spec_name)
            if(spec_name in prods_dict.keys()):
                hash_map[rnum-1,spec_idx] = prods_dict[spec_name] - reacts_dict[spec_name]
                del prods_dict[spec_name]
            else:
                hash_map[rnum-1,spec_idx] = -reacts_dict[spec_name]
        for spec_name in prods_dict.keys():
            spec_idx = specs.index(spec_name)
            hash_map[rnum-1,spec_idx] = prods_dict[spec_name]
    ###
    hash_mask = hash_map!=0
    max_specs = np.amax(np.sum(hash_mask,axis=-1))
    print(np.sum(hash_mask,axis=-1))
    #new_lines.append("%sREAL,DIMENSION(%d,%d) :: PR \n"%(space,nreacts,max_specs))
    ##
    map_lines = []
    coef_lines = []
    ##
    map_line = "%s"%(space)
    coef_line = "%s"%(space)
    for rnum in range(nreacts):
        count = 0
        for L in range(nspec):
            if(hash_mask[rnum,L]):
                count = count + 1
                cstr = "%d,"%(hash_map[rnum,L])
                mstr =  "%d,"%(L+1)
                ##
                if(len(map_line)+len(mstr) >= 72):
                    map_lines.append(map_line+"&\n")
                    map_line = "%s%s"%(space,mstr)
                else:
                    map_line = map_line + mstr
                ###
                if(len(coef_line)+len(cstr) >= 72):
                    coef_lines.append(coef_line+"&\n")
                    coef_line = "%s%s"%(space,cstr)
                else:
                    coef_line = coef_line + cstr
                    
        for rnum in range(max_specs - count):
            cstr = "%d,"%(0)
            mstr =  "%d,"%(nspec)
            ##
            if(len(map_line)+len(mstr) >= 72):
                map_lines.append(map_line+"&\n")
                map_line = "%s%s"%(space,mstr)
            else:
                map_line = map_line + mstr
            ###
            if(len(coef_line)+len(cstr) >= 72):
                coef_lines.append(coef_line+"&\n")
                coef_line = "%s%s"%(space,cstr)
            else:
                coef_line = coef_line + cstr
    ##
    coef_lines.append(coef_line) 
    map_lines.append(map_line) 
    ##
    if(coef_lines[-1][-1]==","):
        line = coef_lines[-1]
        line = line[:-1]+"&\n"
        coef_lines[-1] =  line
    if(map_lines[-1][-1]==","):
        line = map_lines[-1]
        line = line[:-1]+"&\n"
        map_lines[-1] =  line
            
    new_lines.append("%sINTEGER,DIMENSION(%d,%d) :: ST_MAP=RESHAPE((/&\n"\
                    %(space,nreacts,max_specs))
    new_lines = new_lines + map_lines
    line = "%s/),(/%d,%d/))\n"%(space,nreacts,max_specs)
    new_lines.append(line)
    ###
    new_lines.append("%sINTEGER,DIMENSION(%d,%d) :: ST_COEF=RESHAPE((/&\n"\
                    %(space,nreacts,max_specs))
    new_lines = new_lines + coef_lines
    line = "%s/),(/%d,%d/))\n"%(space,nreacts,max_specs)
    new_lines.append(line)
    ##
    new_lines.append("!%sNOTE:LOOP WRITTEN TO REDUCE THE REDUNDANT MULTIPLICATIONS WHILE KEEPING SIMD\n"%(space_m1))
    new_lines.append("!%sNOTE:I COULDN'T THINK OF A BETTER WAY !!\n"%(space_m1))
    new_lines.append("%sDO idx = 1,Np\n"%(space))
    ##
    new_lines.append("%sDO rnum = 1,%d\n"%(space,nreacts))
    new_lines.append("%sDO L = 1,%d\n"%(space,max_specs))
    #new_lines.append("%sPR(rnum,L) = ST_COEF(rnum,L)*RR(idx,rnum)\n"%(space))
    new_lines.append("%sWDOT(idx,ST_MAP(rnum,L)) = WDOT(idx,ST_MAP(rnum,L)) +\n"%(space))
    new_lines.append("%sST_COEF(rnum,L)*RR(idx,rnum)\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND DO\n"%(space))
    ##
    #new_lines.append("!%sWARNING: CAREFUL WHEN PARALLELISING THIS ON GPUS!!\n"%(space_m1))
    #new_lines.append("!%sWARNING: DIFFERENT THREADS WRITING TO SAME LOCATION!!\n"%(space_m1))
    #new_lines.append("%sDO rnum = 1,%d\n"%(space,nreacts))
    #new_lines.append("%sDO L = 1,%d\n"%(space,max_specs))
    #new_lines.append("%sWDOT(idx,ST_MAP(rnum,L)) = WDOT(idx,ST_MAP(rnum,L)) + PR(rnum,L)\n"%(space))
    #new_lines.append("%sEND DO\n"%(space))
    #new_lines.append("%sEND DO\n"%(space))
    ##
    new_lines.append("%sEND DO\n"%(space))
    new_lines.append("%sEND SUBROUTINE\n"%(space))
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    f=open(out_file,"a")
    for l in new_lines:
        f.write(l)
    f.close()
    return
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def write_getrates(inp_file,out_file):
    ###find the skeletal species
    specs = find_sk_species(inp_file)
    nspec = len(specs)
    r_dict = parse_reactions(inp_file,change_order=True)
    nreacts = len(r_dict.keys())
    ##
    new_lines = []
    new_lines.append("%sSUBROUTINE GETRATES_GPU(Np,P, T, Y, ICKWRK, RCKWRK, WDOT)\n"%(space))
    new_lines.append("%sIMPLICIT NONE\n"%(space))
    new_lines.append("%sINTEGER,INTENT(IN) :: Np\n"%(space))
    new_lines.append("%sREAL,INTENT(IN) :: Y(Np,%d),T(Np),P(Np),ICKWRK(*),RCKWRK(*)\n"%(space,nspec))
    new_lines.append("%sREAL,INTENT(OUT) :: WDOT(Np,%d)\n"%(space,nspec))
    new_lines.append("!******************local declerations**************************\n")
    new_lines.append("%sREAL :: logKF(Np,%d),logKR(Np,%d),RR(Np,%d)\n"\
                            %(space,nreacts,nreacts,nreacts))
    new_lines.append("!\n")
#    new_lines.append("%sC(:,:) = Y(:,:)\n"%(space))
    new_lines.append("%sCALL GET_FORWARD_RATE_CONST(Np,%d,T,logKF)\n"%(space,nreacts))
    new_lines.append("%sCALL GET_BACKWARD_RATE_CONST(Np,%d,%d,T,logKF,logKR)\n"%(space,nreacts,nspec))
    new_lines.append("%sCALL GET_NET_REACTION_RATE(Np,logKF,logKR,Y,RR)\n"%(space))
    new_lines.append("%sCALL GET_NET_PRODUCTION_RATE(Np,RR,WDOT)\n"%(space))
    new_lines.append("%sEND SUBROUTINE\n"%(space))
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    new_lines.append("!********************************************************************\n")
    f=open(out_file,"a")
    for l in new_lines:
        f.write(l)
    f.close()
            
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 

if __name__ == "__main__":
    ###outfile
    out_file = os.path.join("./test_gpu","getrates_gpu_reorder.f90")
    ###underlying skeletal mechanism
    inp_file = os.path.join("./test_gpu","chem.ske50.inp")
    ##therm.dat file
    therm_file=os.path.join("./test_gpu","therm.dat")
    ##
    lines = []
    lines.append("!********************************************************************\n")
    lines.append("!********************************************************************\n")
    lines.append("!********************************************************************\n")
    lines.append("!%sWARNING: ALL ARRAYS ARE ASSUMED TO BE IN ROW MAJOR(C)\n")
    lines.append("!********************************************************************\n")
    lines.append("!********************************************************************\n")
    lines.append("!********************************************************************\n")
    ##
    f=open(out_file,"w")
    for l in lines:
        f.write(l)
    f.close()
    
    write_getrates(inp_file,out_file)
    write_forward_rate_const_func(inp_file,out_file)
    write_backward_rate_const_func(inp_file,therm_file,out_file)
    write_net_reaction_rate(inp_file,out_file)
    write_net_production_rate(inp_file,out_file)
