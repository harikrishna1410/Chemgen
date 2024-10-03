##this script adds a species and its reactions to a reduced mechanism
#NOTE: this script does not support the pressure dependent reactions
### But wherever possible I left stubs to implement troe and plog 
import re
import sys
import os
import numpy as np

###this file has the new species you want add and its corresponding reactions
new_file = os.path.join("./test_add_NNH","sk_NNH.inp")
###reduced mechanism you want add this species to
red_file = os.path.join("./test_add_NNH","getrates.f")
###outfile
out_file = os.path.join("./test_add_NNH","getrates_add_NNH.f")
###underlying skeletal mechanism
sk_file = os.path.join("./test_gpu","chem.ske50.inp")

###file containing qssa species
qssa_spec = os.path.join("./test_add_NNH","qssa_spec.txt")

##therm.dat file
therm_file=os.path.join("./test_gpu","therm.dat")
###QSSA flag: flag to indicate if the new species should be put in qss
qssa_flag = True
##*****************************stuff starts here***********************************
##*****************************stuff starts here***********************************
##*****************************stuff starts here***********************************
###element weights
####the new species should have only these species
###add more to this list. For JICF H2 case these were
###enough
elem_wt = {}
elem_wt["H"] = 1.007969975471497E0
elem_wt["O"] = 1.599940013885498E1
elem_wt["C"] = 1.201115036010742E1
elem_wt["N"] = 1.400669956207276E1

###Value of universal gas constant in cal/mol/k
##I back calculated this from the JICF mechanism from tianfeng
R_c = 1.987215575926745
#****************************************************************
#****************************************************************
#****************************************************************
####function to check if the species name is valid. i.e if it only 
###has elements from elem_wt dict keys
def check_species_validity(spec_name):
    temp_spec = spec_name.strip()
    ##just replace all the elements with empty string
    for elem in elem_wt.keys():
        temp_spec = temp_spec.replace(elem,"")
    ###now deal with singlet (s)
    temp_spec = temp_spec.replace("(s)","")
    temp_spec = temp_spec.replace("(S)","")
    ###now this should either be an int or empty
    if(temp_spec == "" or re.fullmatch("[0-9]*",temp_spec) is not None):
        return True
    else:
        return False
#****************************************************************
#****************************************************************
#****************************************************************
###function to parse the species name are return the elements
def parse_species_name(spec_name):
    elem_comp = {}
    if(not check_species_validity(spec_name)):
        print("Error: not a valid species!!!")
        sys.exit()
    for elem in elem_wt.keys():
        l = re.findall("%s[0-9]*"%(elem),spec_name)
        nelem=[]
        for i in l:
            if(i==elem):
                nelem.append(1)
            else:
                nelem.append(int(i.split(elem)[1]))
        elem_comp[elem] = sum(nelem)
    return elem_comp
#****************************************************************
#****************************************************************
#****************************************************************
##get molecular weight
def get_mol_wt(spec_name):
    elem_comp = parse_species_name(spec_name)
    mol_wt = 0.0
    for elem in elem_wt.keys():
        mol_wt = mol_wt + elem_wt[elem]*elem_comp[elem]
    return mol_wt
#****************************************************************
#****************************************************************
#****************************************************************
#function to parse all the reactions
###order_change = True puts all standard,falloff,third-body,plog reactions
###in that order
def parse_reactions(in_file,change_order=False):
    f=open(in_file,"r")
    all_lines=f.readlines()
    f.close()
    ###
    in_react = False
    react_lines = []
    for l in all_lines:
        if("REACTIONS" in l):
            in_react = True
        else:
            if(in_react):
                if("END" == l.strip()):
                    break
                ###this removes all the comments
                react_lines.append(l.strip().split("!")[0])
    react_lines = [i for i in react_lines if i != ""]
    ##I could create a seperate reaction class but didn't have time 
    ##to do this. I simply used dict of dicts
    ###          rnum
    ##        /   |   \
    ###      eqn arh  troei plog third-body
    r_dict = {}
    rnum = 0
    plog_r = []
    troe_r = []
    third_r = []
    for l in react_lines:
        if(l[0] == "!"):
            continue
        if("=" in l):
            print(l)
            rnum = rnum + 1
            r_dict[rnum] = {}
            temp_list = [i for i in l.strip().replace("\t"," ").split(" ") if i!=""]
            r_dict[rnum]["eqn"] = temp_list[0]
            r_dict[rnum]["arh"] = [float(i) for i in temp_list[1:]]
        ### I am assuming either troe or plog
        ### I am just adding these as strings when needed I will use these
        ### I am lucky that NNH reactions for H2 JICF doesn't have any of these!!!
        elif("LOW" in l):
            r_dict[rnum]["troe"] = {}
            temp_list = [i for i in l.strip().replace("\t"," ").split(" ") if i!=""]
            r_dict[rnum]["troe"]["low"] = [float(i) for i in temp_list[1:-1]]
        elif("TROE" in l):
            temp_list = [i for i in l.strip().replace("\t"," ").split(" ") if i!=""]
            r_dict[rnum]["troe"]["troe"] = [float(i) for i in temp_list[1:-1]]
            troe_r.append(rnum)
        elif("PLOG" in l):
            if ("PLOG" in r_dict[rnum].keys()):
                num = len(r_dict[rnum]["PLOG"].keys())
                r_dict[rnum]["PLOG"][num+1] = l.strip().split()[1:-1]
            else:
                r_dict[rnum]["PLOG"] = {}
                r_dict[rnum]["PLOG"][1] = l.strip().split()[1:-1]
                plog_r.append(rnum)
        elif("DUP" in l):
            r_dict[rnum]["DUP"] = True
        ###I am just assuming this will be a third body collision efficiency line
        else:
            r_dict[rnum]["third-body"] = {}
            temp_list = [i.strip() for i in l.split("/")[:-1]]
            for i in range(len(temp_list)//2):
                if(float(temp_list[2*i+1])-1.0 != 0.0):
                    r_dict[rnum]["third-body"][temp_list[2*i]] = float(temp_list[2*i+1])-1.0
            if(rnum not in troe_r):
                third_r.append(rnum)
    ###
    if(not change_order):
        return r_dict
    else:
        nreacts = len(r_dict.keys())
        n_third = len(third_r)
        n_plog = len(plog_r)
        n_troe = len(troe_r)
        n_st = nreacts - n_third - n_plog - n_troe
        new_rnum = 1
        new_r_dict = {}
        print("st reacts:",n_st)
        print("troe reacts:",n_troe)
        print("third reacts:",n_third)
        print("plog reacts:",n_plog)
        for i in range(len(r_dict.keys())):
            if(i+1 in troe_r):
                rnum = n_st+troe_r.index(i+1)+1
                if(rnum in new_r_dict.keys()):
                    print("RPARSE ERROR: %d is already there"%(rnum))
                    sys.exit()
                new_r_dict[rnum] = r_dict.pop(i+1)
            elif(i+1 in third_r):
                rnum = n_st+n_troe+third_r.index(i+1)+1
                if(rnum in new_r_dict.keys()):
                    print("RPARSE ERROR: %d is already there"%(rnum))
                    sys.exit()
                new_r_dict[rnum] = r_dict.pop(i+1)
            elif(i+1 in plog_r):
                rnum = n_st+n_troe+n_third+plog_r.index(i+1)+1
                if(rnum in new_r_dict.keys()):
                    print("RPARSE ERROR: %d is already there"%(rnum))
                    sys.exit()
                new_r_dict[rnum] = r_dict.pop(i+1)
            else:
                new_r_dict[new_rnum] = r_dict.pop(i+1)
                new_rnum = new_rnum + 1
        return new_r_dict,n_st,n_troe,n_third,n_plog
#****************************************************************
#****************************************************************
#****************************************************************
###function to return the lines of a subroutine
def find_subroutine_lines(name,all_lines):
    ret_lines = []
    in_sub = False
    count = 0
    for l in all_lines:
        if in_sub:
            ret_lines.append(l)
            if (l.strip()=="END"):
                end = count
                return ret_lines,start,end
        else:
            if re.match('SUBROUTINE\s*%s'%(name), l.strip()):
                start = count
                ret_lines.append(l)
                in_sub = True
        count = count + 1
    return ret_lines,start,end
#****************************************************************
#****************************************************************
#****************************************************************
##returns all the species in .inp chemkin file
def find_sk_species(inp_file):
    f=open(inp_file)
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
    return spec_list
#****************************************************************
#****************************************************************
#****************************************************************
###function to return the transported species in input .f file
def find_red_species():
    sk_spec = find_sk_species(sk_file)
    f=open(qssa_spec,"r")
    qssa_spec_list = [i.strip() for i in f.readlines()]
    f.close()
    
    return [i for i in sk_spec if i not in qssa_spec_list]

#****************************************************************
#****************************************************************
#****************************************************************
###given the reaction name returns the products and reactants
###also returns its stoichiometric coefficients
def get_prod_reacts(rname,new_file=None,remove_dups=True):
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
    rname = rname.replace("(+M)","")
    rname = rname.replace("+M","")
    ###
    reacts_temp = [i.strip() for i in rname.split(sym)[0].split("+")]
    prods_temp =  [i.strip() for i in rname.split(sym)[1].split("+")]
    ##
    sk_specs = find_sk_species(sk_file)
    if(new_file is not None):
        new_specs = find_sk_species(new_file)
    else:
        new_specs = []
    all_specs = sk_specs+new_specs
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
#****************************************************************
#****************************************************************
#****************************************************************
###function returns the nasa polynomila coeff and temeparture limits
def get_nasa_poly(spec_name,in_file):
    coef = []
    temp_limits = []
    with open(in_file, "r") as filestream:
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
    print(spec_name,temp_limits)
    return temp_limits,coef
    
#****************************************************************
#****************************************************************
#****************************************************************
###funtion to add smh for a species
def add_smh(spec_name,in_file,out_file):
    ###get the coefficients
    temp_limits,coef=get_nasa_poly(spec_name,therm_file)
    high_coef = coef[0:7] 
    low_coef = coef[7:14]
    ###find the skeletal species
    sk_species = find_sk_species(sk_file)
    nspec = len(sk_species)
    ###
    space = "      "
    extra_space = "         "
    new_lines = []
    new_lines.append("C %s\n"%(spec_name))
    ####(s/R-H/RT) = a7 - a6/T + a1*(logT - 1) + a2/2*T + a3/6*(T^2) 
    ####+ a4/12*(T^3) + a5/20*(T^4)
    T = ["","*TI","*TN(1)","*TN(2)","*TN(3)","*TN(4)","*TN(5)"]
    new_lines.append("%sIF (T .GT. %d) THEN\n"%(space,temp_limits[-1]))
    line = "%sSMH(%d) = "%(space,nspec+1)
    count = 0
    ###reorder the coeff to writing order and divide
    write_coef = [high_coef[6],-high_coef[5],high_coef[0],high_coef[1]/2
                 ,high_coef[2]/6,high_coef[3]/12,high_coef[4]/20]
    for idx,a in enumerate(write_coef):
        if(np.abs(a) > 0):
            if(a > 0):
                sign = "+"
            else:
                sign = "-"
            line = line + "%s%21.15E%s"%(sign,np.abs(a),T[idx])
            count = count + 1
        if(count>0 and count%2 == 0):
            line = line + "\n     *%s"%(extra_space)
    ###append line
    line = line + "\n"
    new_lines.append(line.replace("E","D"))
    ###writing the low
    new_lines.append("%sELSE\n"%(space))
    line = "%sSMH(%d) = "%(space,nspec+1)
    ###reorder the coeff to writing order and divide
    write_coef = [low_coef[6],-low_coef[5],low_coef[0],low_coef[1]/2
                 ,low_coef[2]/6,low_coef[3]/12,low_coef[4]/20]
    count = 0
    for idx,a in enumerate(write_coef):
        if(np.abs(a) > 0):
            if(a > 0):
                sign = "+"
            else:
                sign = "-"
            line = line + "%s%21.15E%s"%(sign,np.abs(a),T[idx])
            count = count + 1
        if(count>0 and count%2 == 0):
            line = line + "\n     *%s"%(extra_space)
    line = line + "\n"
    new_lines.append(line.replace("E","D"))
    new_lines.append("%sENDIF\n"%(space))
    ###write the lines
    f=open(in_file,"r")
    all_lines = f.readlines()
    rdsmh_lines,start,end = find_subroutine_lines("RDSMH",all_lines)
    
    new_lines.reverse()
    for l in new_lines:
        all_lines.insert(end,l)
    f=open(out_file,"w")
    for l in all_lines:
        f.write(l)
    f.close()
    return
#****************************************************************
#****************************************************************
#****************************************************************
###add the new reactions to ratt
###react_file: file in ck format containing the new species
###sk_file: skeletal mech of the reduced mech you want to add species to
###in_file: reduced mech file
###out_file: output file
def add_ratt(react_file,sk_file,in_file,out_file):
    ##
    ##old_spec = find_sk_species("./test_add_NNH/chem.ske50.inp")
    old_spec = find_sk_species(sk_file)
    old_r_dict = parse_reactions(sk_file)
    nspec = len(old_spec)
    ##
    new_spec = find_sk_species(react_file)
    new_r_dict = parse_reactions(react_file)
    if(len(new_spec)>1):
        print("ERROR: can only add 1 spec at a time")
        sys.exit()
    ### check if all reactions are valid
    print("testing reaction valididty")
    for rnum in new_r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(new_r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        for spec in reacts+prods:
            if(spec not in new_spec and spec not in old_spec):
                print("ERROR: %s is not a valid species"%(spec))
                sys.exit()
    space = "      "
    new_lines = []
    new_lines.append("%sEG(%d) = EXP(SMH(%d))\n"%(space,nspec+1,nspec+1))
    old_nreacts = len(old_r_dict.keys())
    count = 0
    for rnum in new_r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(new_r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        if("troe" in new_r_dict[rnum].keys() or "PLOG" in new_r_dict[rnum].keys()\
            or "third-body" in new_r_dict[rnum].keys()):
            print("WARNING: %s is pressure dependent"%(new_r_dict[rnum]["eqn"]))
            print("reaction dict keys ",new_r_dict[rnum].keys())
            print("skipping it!!")
            continue
        print("adding reaction %s"%(new_r_dict[rnum]["eqn"]))
        count = count + 1
        ###reaction eqn line
        new_lines.append("C     R%d: %s\n"%(count+old_nreacts,new_r_dict[rnum]["eqn"]))
        ###add RF
        line = "%sRF(%d) = EXP("%(space,count+old_nreacts)
        A = new_r_dict[rnum]["arh"][0]
        beta = new_r_dict[rnum]["arh"][1]
        Ea = new_r_dict[rnum]["arh"][2]
        ###log(A)
        line = line + ("%21.15E"%(np.log(A))).replace("E","D")
        ###beta*log(T)
        if(np.abs(beta) > 0):
            if(beta>0):
                sign = "+"
            else:
                sign = "-"
            line = line + (" %s%21.15E*ALOGT"%(sign,np.abs(beta)))\
                  .replace("E","D")
            line = line+"\n"
        ###-Ea/R_cT (R_c is defined above)
        if(np.abs(Ea) > 0):
            if(np.abs(beta)>0):
                line = line + "     *"
            if(Ea>0):
                sign="-"
            else:
                sign="+"
            line = line + (" %s%21.15E*TI"%(sign,np.abs(Ea)/R_c))\
                          .replace("E","D")
        line = line + ")\n"
        new_lines.append(line)
        ###*****add EQK
        ##
        all_specs = old_spec+new_spec
        ###prods
        for i in range(prods_dict[prods[0]]):
            if(i == 0):
                prod_EG = "EG(%d)"%(all_specs.index(prods[0])+1)
            else:
                prod_EG = prod_EG + "*EG(%d)"%(all_specs.index(prods[0])+1)
        if(len(prods)>1):
            for p in prods[1:]:
                for i in range(prods_dict[p]):
                    prod_EG = prod_EG + "*EG(%d)"%(all_specs.index(p)+1)
        ##reacts
        for i in range(reacts_dict[reacts[0]]):
            if(i == 0):
                react_EG = "EG(%d)"%(all_specs.index(reacts[0])+1)
            else:
                react_EG = react_EG + "/EG(%d)"%(all_specs.index(reacts[0])+1)
        if(len(reacts)>1):
            for p in reacts[1:]:
                for i in range(reacts_dict[p]):
                    react_EG = react_EG + "/EG(%d)"%(all_specs.index(p)+1)
        ##add PFAC
        exp = 0
        for key in reacts_dict.keys():
            exp = exp - reacts_dict[key]
        for key in prods_dict.keys():
            exp = exp + prods_dict[key]
        if(exp < 0):
            if(exp == -1):
                pfac_str = "/PFAC1"
            elif(exp == -2):
                pfac_str = "/PFAC2"
            elif(exp == -3):
                pfac_str = "/PFAC3"
            else:
                print("ERROR: I only know 1,2, and 3")
                sys.exit()
        elif(exp > 0):
            if(exp == 1):
                pfac_str = "*PFAC1"
            elif(exp == 2):
                pfac_str = "*PFAC2"
            elif(exp == 3):
                pfac_str = "*PFAC3"
            else:
                print("ERROR: I only know 1,2, and 3")
                sys.exit()
        else:
            pfac_str = ""
        line = "%sEQK = %s/%s%s\n"%(space,prod_EG,react_EG,pfac_str)
        new_lines.append(line)
        ###
        line = "%sRB(%d) = RF(%d) / MAX(EQK,SMALL)\n"%(space,count+old_nreacts\
                                                    ,count+old_nreacts)
        new_lines.append(line)
        
    ###read lines 
    f=open(in_file,"r")
    all_lines=f.readlines()
    f.close()
    ###I have all new lines now just put it in right place
    ratt_lines,start,end=find_subroutine_lines("RATT",all_lines)
    for idx in range(start,end+1):
        if(re.match(r"\s*DIMENSION\s*SMH\([0-9]*\),\s*EG\([0-9]*\)",all_lines[idx])):
            all_lines[idx] = all_lines[idx].replace("SMH(%d)"%(nspec)\
                            ,"SMH(%d)"%(nspec+1)).replace("EG(%d)"%(nspec),\
                              "EG(%d)"%(nspec+1))
        if("EG(%d) = EXP(SMH(%d))"%(nspec,nspec) in all_lines[idx]):
            all_lines.insert(idx+1, new_lines[0])
        if("RB(%d)"%(old_nreacts) in all_lines[idx]):
            new_lines.reverse()
            for l in new_lines[:-1]:
                all_lines.insert(idx+1,l)
            break
    ##write everything
    f=open(out_file,"w")
    for l in all_lines:
        f.write(l)
    f.close()
    ##
    return 
#****************************************************************
#****************************************************************
#****************************************************************
###add the new reactions to ratx
###react_file: file in ck format containing the new species
###sk_file: skeletal mech of the reduced mech you want to add species to
###in_file: reduced mech file
###out_file: output file
def add_ratx(react_file,sk_file,in_file,out_file,qssa_flag):
    old_spec = find_sk_species(sk_file)
    old_r_dict = parse_reactions(sk_file)
    old_nreacts = len(old_r_dict.keys())
    nspec = len(old_spec)
    ##
    new_spec = find_sk_species(react_file)
    new_r_dict = parse_reactions(react_file)
    ##
    red_spec = find_red_species()
    old_red_nspec = len(red_spec)
    if(not qssa_flag):
        red_spec = red_spec + new_spec
    # else:
    #     print("Sorry, for now I don't support qssa species")
    #     print("But, soon :-)")
    #     sys.exit()
    ###
    space = "      "
    new_lines = []
    count = 0
    for rnum in new_r_dict.keys(): 
        reacts_dict,prods_dict=get_prod_reacts(new_r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        ##
        if("troe" in new_r_dict[rnum].keys() or "PLOG" in new_r_dict[rnum].keys()\
            or "third-body" in new_r_dict[rnum].keys()):
            print("WARNING: %s is pressure dependent"%(new_r_dict[rnum]["eqn"]))
            print("reaction dict keys ",new_r_dict[rnum].keys())
            print("skipping it!!")
            continue
        ##
        print("adding reaction %s"%(new_r_dict[rnum]["eqn"]))
        count = count + 1
        new_lines.append("C     R%d: %s\n"%(count+old_nreacts,new_r_dict[rnum]["eqn"]))
        line = "%sRF(%d) = RF(%d)"%(space,count+old_nreacts,count+old_nreacts)
        ###reacts
        for r in reacts:
            if(r in red_spec):
                for i in range(reacts_dict[r]):
                    line = line + "*C(%d)"%(red_spec.index(r)+1)
        line = line + "\n"
        new_lines.append(line)
        ##prods
        line = "%sRB(%d) = RB(%d)"%(space,count+old_nreacts,count+old_nreacts)
        ###reacts
        for r in prods:
            if(r in red_spec):
                for i in range(prods_dict[r]):
                    line = line + "*C(%d)"%(red_spec.index(r)+1)
        line = line + "\n"
        new_lines.append(line)
    ##
    ###read lines 
    f=open(in_file,"r")
    all_lines=f.readlines()
    f.close()
    ###I have all new lines now just put it in right place
    ratt_lines,start,end=find_subroutine_lines("RATX",all_lines)
    for idx in range(start,end+1):
        if("DO K" in all_lines[idx]):
            all_lines[idx] = all_lines[idx].replace("%d"%(old_red_nspec)\
                                               ,"%d"%(len(red_spec)))
        if("RB(%d)"%(old_nreacts) in all_lines[idx]):
            new_lines.reverse()
            for l in new_lines:
                all_lines.insert(idx+1,l)
            break
    ##write everything
    f=open(out_file,"w")
    for l in all_lines:
        f.write(l)
    f.close()
    return
#****************************************************************
#****************************************************************
#****************************************************************
###function add reactions to RDOT subroutine
def add_rdot(react_file,sk_file,in_file,out_file,qssa_flag):
    ###
    old_spec = find_sk_species(sk_file)
    old_r_dict = parse_reactions(sk_file)
    old_nreacts = len(old_r_dict.keys())
    nspec = len(old_spec)
    ##
    new_spec = find_sk_species(react_file)
    new_r_dict = parse_reactions(react_file)
    ###
    red_spec = find_red_species()
    old_red_nspec = len(red_spec)
    if(not qssa_flag):
        red_spec = red_spec + new_spec
    new_red_nspec = len(red_spec)
    ##
    space = "      "
    new_lines = []
    count=0
    for rnum in new_r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(new_r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        ##
        if("troe" in new_r_dict[rnum].keys() or "PLOG" in new_r_dict[rnum].keys()\
            or "third-body" in new_r_dict[rnum].keys()):
            print("WARNING: %s is pressure dependent"%(new_r_dict[rnum]["eqn"]))
            print("reaction dict keys ",new_r_dict[rnum].keys())
            print("skipping it!!")
            continue
        ###
        print("adding reaction %s"%(new_r_dict[rnum]["eqn"]))
        count = count + 1
        line = "%sROP = RF(%d) - RB(%d)\n"%(space,count+old_nreacts,count+old_nreacts)
        print("line:",line)
        new_lines.append(line)
        ##reactants
        for r in reacts:
            if(r in red_spec):
                if(reacts_dict[r]>1):
                    st_coef="%d*"%(reacts_dict[r])
                else:
                    st_coef = ""
                line = "%sWDOT(%d) = WDOT(%d) - %sROP\n"\
                  %(space,red_spec.index(r)+1,red_spec.index(r)+1,st_coef)
                new_lines.append(line)
        ##products 
        for r in prods:
            if(r in red_spec):
                if(prods_dict[r]>1):
                    st_coef="%d*"%(prods_dict[r])
                else:
                    st_coef = ""
                line = "%sWDOT(%d) = WDOT(%d) + %sROP\n"\
                  %(space,red_spec.index(r)+1,red_spec.index(r)+1,st_coef)
                new_lines.append(line)
    f=open(in_file,"r")
    all_lines = f.readlines()
    f.close()
    ###
    rdot_lines,start,end=find_subroutine_lines("RDOT",all_lines)
    new_lines.reverse()
    for l in new_lines:
        all_lines.insert(end,l)
    f=open(out_file,"w")
    for l in all_lines:
        if("DO K" in l):
            l = l.replace(" %d"%(old_red_nspec)," %d"%(new_red_nspec))
        f.write(l)
    f.close()
    return
#****************************************************************
#****************************************************************
#****************************************************************
def modify_getrates(react_file,sk_file,in_file,out_file,qssa_flag):
    ###
    old_spec = find_sk_species(sk_file)
    old_r_dict = parse_reactions(sk_file)
    old_nreacts = len(old_r_dict.keys())
    old_nspec = len(old_spec)
    ##
    new_spec = find_sk_species(react_file)
    new_r_dict = parse_reactions(react_file)
    ###
    red_spec = find_red_species()
    old_red_nspec= len(red_spec)
    if(not qssa_flag):
        red_spec = red_spec + new_spec
    ###
    red_nspec= len(red_spec)
    ###
    count = 0
    for rnum in new_r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(new_r_dict[rnum]["eqn"])
        ##
        if("troe" in new_r_dict[rnum].keys() or "PLOG" in new_r_dict[rnum].keys()\
            or "third-body" in new_r_dict[rnum].keys()):
            continue
        count= count + 1
    n_reacts = count + old_nreacts
    ##
    f=open(in_file,"r")
    all_lines=f.readlines()
    f.close()
    
    getrates_lines,start,end = find_subroutine_lines("GETRATES",all_lines)
    for i in range(start,end+1):
        if("RF(%d)"%(old_nreacts) in all_lines[i]):
            all_lines[i]=all_lines[i].replace("RF(%d)"%(old_nreacts),\
           "RF(%d)"%(n_reacts))
        ###
        if("RB(%d)"%(old_nreacts) in all_lines[i]):
            all_lines[i]=all_lines[i].replace("RB(%d)"%(old_nreacts),\
           "RB(%d)"%(n_reacts))
        ###
        if("C(%d)"%(old_red_nspec) in all_lines[i]):
            all_lines[i]=all_lines[i].replace("C(%d)"%(old_red_nspec),\
           "C(%d)"%(red_nspec))
        ###
        if(qssa_flag):
            if("XQ(%d)"%(old_nspec-old_red_nspec) in all_lines[i]):
                all_lines[i]=all_lines[i].replace("XQ(%d)"%(old_nspec-old_red_nspec),\
               "XQ(%d)"%(old_nspec-old_red_nspec+1))
    ### 
    f=open(out_file,"w")
    for l in all_lines:
        f.write(l)
    f.close()
    return
#****************************************************************
#****************************************************************
#****************************************************************
def add_ytcp(react_file,sk_file,in_file,out_file):
    old_spec = find_sk_species(sk_file)
    old_r_dict = parse_reactions(sk_file)
    old_nreacts = len(old_r_dict.keys())
    old_nspec = len(old_spec)
    ##
    new_spec = find_sk_species(react_file)
    new_r_dict = parse_reactions(react_file)
    ###
    red_spec = find_red_species()
    old_red_nspec= len(red_spec)
    ###
    n_reacts = len(new_r_dict.keys()) + old_nreacts
    ###
    f=open(in_file,"r")
    all_lines = f.readlines()
    f.close()
    ###
    ytcp_lines,start,end = find_subroutine_lines("YTCP",all_lines)
    for i in range(start,end+1):
        if("C(%d)"%(old_red_nspec) in all_lines[i]):
            mol_wt = get_mol_wt(new_spec[0])
            mol_wt_str = ("%21.15E"%(mol_wt)).replace("E","D")
            line = "      C(%d) = Y(%d)/%s\n"%(old_red_nspec+1,\
                            old_red_nspec+1,mol_wt_str)
            all_lines.insert(i+1,line)
        if("DO K" in all_lines[i]):
            all_lines[i] = all_lines[i].replace("%d"%(old_red_nspec)\
                                               ,"%d"%(old_red_nspec+1))
    f=open(out_file,"w")
    for l in all_lines:
        f.write(l)
    f.close()
    return 
#****************************************************************
#****************************************************************
#****************************************************************
def add_qssa(react_file,sk_file,in_file,out_file,qssa_file):
    old_spec = find_sk_species(sk_file)
    old_r_dict = parse_reactions(sk_file)
    old_nreacts = len(old_r_dict.keys())
    old_nspec = len(old_spec)
    ##
    new_spec = find_sk_species(react_file)
    new_r_dict = parse_reactions(react_file)
    ###
    red_spec = find_red_species()
    old_red_nspec= len(red_spec)
    ###
    n_reacts = len(new_r_dict.keys()) + old_nreacts
    ###
    f=open(qssa_file,"r")
    qssa_spec_list = [i.strip() for i in f.readlines()]
    f.close()
    ###init new lines
    space = "      "
    ###RHS lines
    b_line = "%sA%d_0 = ("%(space,len(qssa_spec_list)+1)
    b_line_len = 12
    ##coupling lines 
    c_lines = []
    for idx in range(1,len(qssa_spec_list)+1):
        c_lines.append("%sA%d_%d = ("%(space,len(qssa_spec_list)+1,idx))
    ##diagonal line
    d_line = "%sDEN = "%(space)
    d_line_len = 12
    ###zero lines
    z_lines = []
    ###coupling array to keep how this new species 
    ##is coupled with the existing ones
    ##array shape is (n_qssa_spec (excluding the new one),new num reacts)
    ##1 indicates new spec is on the product side and index species is on the 
    ##reactant side
    ##-1 is the opposite
    coupling_arr = np.zeros((len(qssa_spec_list),len(new_r_dict.keys())),\
                            dtype=np.int32)
    ###these are the final lines 
    xq_lines = []
    ##
    rr_lines = []
    for rnum in new_r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(new_r_dict[rnum]["eqn"])
        reacts = reacts_dict.keys()
        prods = prods_dict.keys()
        ###
        if(new_spec[0] in reacts):
            couple_flag = False
            for r in reacts:
                if(r in qssa_spec_list):
                    z_lines.append("%sRF(%d) = 0.0D0\n"%(space,rnum+old_nreacts))
            for r in prods:
                if(r in qssa_spec_list):
                    couple_flag = True
                    couple_spec = r
            if(couple_flag):
                idx = qssa_spec_list.index(couple_spec)
                coupling_arr[idx,rnum-1] = -1
                if(len(c_lines[idx])<65):
                    c_lines[idx] = c_lines[idx] + " +RB(%d)"%(rnum+old_nreacts)
                else:
                    c_lines[idx] = c_lines[idx] + "\n     *  +RB(%d)"\
                        %(rnum+old_nreacts)
            else:
                if(b_line_len<60):
                    ###RHS
                    b_line = b_line + " +RB(%d)"%(rnum+old_nreacts)
                    b_line_len += 8
                else:
                    ###RHS
                    b_line = b_line + "\n     *  +RB(%d)"%(rnum+old_nreacts)
                    b_line_len = 15
            if(d_line_len<60):
                ###diagonal line
                d_line = d_line + " +RF(%d)"%(rnum+old_nreacts)
                d_line_len += 8
            else:
                ###diagonal line
                d_line = d_line + "\n     *  +RF(%d)"%(rnum+old_nreacts)
                d_line_len = 15
        else:
            couple_flag = False
            for r in prods:
                if(r in qssa_spec_list):
                    z_lines.append("%sRB(%d) = 0.0D0\n"%(space,rnum+old_nreacts))
            for r in reacts:
                if(r in qssa_spec_list):
                    couple_flag = True
                    couple_spec = r
            if(couple_flag):
                idx = qssa_spec_list.index(couple_spec)
                coupling_arr[idx,rnum-1] = 1
                if(len(c_lines[idx])<65):
                    c_lines[idx] = c_lines[idx] + " +RF(%d)"%(rnum+old_nreacts)
                else:
                    c_lines[idx] = c_lines[idx] + "\n     *  +RF(%d)"\
                        %(rnum+old_nreacts)
            else:
                ###RHS
                if(b_line_len<60):
                    b_line = b_line + " +RF(%d)"%(rnum+old_nreacts)
                    b_line_len += 8
                else:
                    b_line = b_line + "\n     *  +RF(%d)"%(rnum+old_nreacts)
                    b_line_len = 15
            ###diagonal line
            if(d_line_len<60):
                d_line = d_line + " +RB(%d)"%(rnum+old_nreacts)
                d_line_len += 8
            else:
                d_line = d_line + "\n     *  +RB(%d)"%(rnum+old_nreacts)
                d_line_len = 15
    
    ###ADD den to new_spec coefficients 
    d_line = d_line + "\n"
    if(b_line_len<60):
        b_line = b_line + ")/DEN\n"
    else:
        b_line = b_line + "\n     *  )/DEN\n"
    ##
    for idx in range(len(c_lines)):
        if(c_lines[idx] != "%sA%d_%d = ("%(space,len(qssa_spec_list)+1\
                                               ,idx+1)):
            if(len(c_lines[idx])<65):
                print(c_lines[idx],idx)
                c_lines[idx] =  c_lines[idx] + ")/DEN\n"
            else:
                c_lines[idx] =  c_lines[idx] + "\n     *  )/DEN\n"
    ###now check if the coupling is with more than 1 species 
    print(coupling_arr)
    idxs = np.array(np.nonzero(np.abs(coupling_arr)))
    if(np.amax(idxs[0])!=np.amin(idxs[0])):
        print("Sorry can't deal with species coupled with more than 1 species ")
        print("use make-remove_spec_qssa_with_coupling.py script")
        sys.exit()
    ##lines to be added in the gauss elimination part
    e_lines = []
    ##now that the new species is only coupled with one other species 
    ##its easy to handle this. I don't have to figure out how tianfeng 
    ##is doing his elimination
    spec_idx = idxs[0,0]+1
    new_spec_idx = len(qssa_spec_list) + 1
    line = "%sA%d_0 = A%d_0 + A%d_0*A%d_%d\n"\
        %(space,spec_idx,spec_idx,new_spec_idx,spec_idx,new_spec_idx)
    e_lines.append(line)
    line = "%sDEN = 1.0 - A%d_%d*A%d_%d\n"%\
            (space,spec_idx,new_spec_idx,new_spec_idx,spec_idx)
    e_lines.append(line)
    ###figure out all the coefficients the spec_idx species has 
    ##also modify the lines
    f=open(in_file,"r")
    all_lines=f.readlines()
    f.close()
    qss_lines,start,end = find_subroutine_lines("QSSA", all_lines)
    spec_name = qssa_spec_list[spec_idx-1].strip()
    in_flag = False
    coefs=[]
    idx = start
    while idx <= end:
        line = all_lines[idx]
        # print(in_flag)
        #print(line)
        if(line.strip() == "C     %s"%(spec_name)):
            in_flag = True
        if(in_flag):
            ###reached the next species break
            if(line.strip() == "C     %s"%(qssa_spec_list[spec_idx].strip())):
                # print("inserting coeffcient lines")
                ##insert new coefficient here
                new_line = "%sA%d_%d = ("%(space,spec_idx,len(qssa_spec_list)+1)
                for rnum in new_r_dict.keys():
                    if(coupling_arr[spec_idx-1,rnum-1]==1):
                        new_line = new_line + "+RB(%d)"%\
                            (rnum+old_nreacts)
                    elif(coupling_arr[spec_idx-1,rnum-1]==-1):
                        new_line = new_line + "+RF(%d)"%\
                                (rnum+old_nreacts)
                new_line = new_line + ")/DEN\n"
                all_lines.insert(idx, new_line)
                break
            if("=" in line):
                if("DEN =" in line):
                    if("*" in all_lines[idx+1]):
                        while ("*" in all_lines[idx+1]):
                            idx = idx + 1
                    new_line = ""
                    for rnum in new_r_dict.keys():
                        if(coupling_arr[spec_idx-1,rnum-1]==1):
                            new_line = new_line + "+RF(%d)"%\
                                (rnum+old_nreacts)
                        elif(coupling_arr[spec_idx-1,rnum-1]==-1):
                                new_line = new_line + "+RB(%d)"%\
                                (rnum+old_nreacts)
                    new_line = new_line + "\n"
                    all_lines[idx] = all_lines[idx].rstrip("\n") + new_line
                    # print(new_line)
                else:
                    coefs.append(line.split("=")[0].strip())
        idx = idx + 1
    ##
    for c in coefs:
        e_lines.append("%s%s = %s/DEN\n"%(space,c,c))
    idx = start
    in_flag = False
    ins_flag=True
    while idx <= end:
        line = all_lines[idx]
        if(line.strip() == "C     %s"%(qssa_spec_list[-1].strip())):
            in_flag = True
        if(in_flag):
            if(line.strip()=="C"):
                e_lines.reverse()
                for l in e_lines:
                    all_lines.insert(idx,l)
                all_lines.insert(idx,"C\n")
                c_lines.reverse()
                for l in c_lines:
                    if(l[-1] != "("):
                        print(l)
                        all_lines.insert(idx,l)
                all_lines.insert(idx,b_line)
                all_lines.insert(idx, d_line)
                all_lines.insert(idx,"C     %s\n"%(new_spec[0]))
                in_flag=False
        else:
            ###insert the xq line
            if(re.match(r"\s*RF\(\s*[0-9]*\)\s*=\s*RF",line)):
                new_line = "%sXQ(%d) = A%d_0 + A%d_%d*XQ(%d)\n"\
                    %(space,len(qssa_spec_list)+1,len(qssa_spec_list)+1,\
                      len(qssa_spec_list)+1,spec_idx,spec_idx)
                if(ins_flag):
                    all_lines.insert(idx,"C\n")
                    all_lines.insert(idx,new_line)
                    ins_flag = False
            if(line.strip()=="END"):
                all_lines.insert(idx,"C\n")
                all_qss = qssa_spec_list+new_spec
                r_list =list(new_r_dict.keys())
                r_list.reverse()
                for rnum in r_list:
                    reacts_dict,prods_dict=get_prod_reacts(new_r_dict[rnum]["eqn"])
                    reacts = reacts_dict.keys()
                    prods = prods_dict.keys()
                    #
                    for r in reacts:
                        if(r in all_qss):
                            flag=True
                            for l in z_lines:
                                if("RF(%d)"%(rnum+old_nreacts) in l):
                                    flag = False
                                    break
                            if(flag):
                                new_line = "%sRF(%d) = RF(%d)*XQ(%d)\n"%\
                                (space,rnum+old_nreacts,rnum+old_nreacts,\
                                 all_qss.index(r)+1)
                                all_lines.insert(idx,new_line)
                                print(new_line)
                    ###
                    for r in prods:
                        if(r in all_qss):
                            flag=True
                            for l in z_lines:
                                if("RB(%d)"%(rnum+old_nreacts) in l):
                                    flag = False
                                    break
                            if(flag):
                                new_line = "%sRB(%d) = RB(%d)*XQ(%d)\n"%\
                                (space,rnum+old_nreacts,rnum+old_nreacts,\
                                 all_qss.index(r)+1)
                                all_lines.insert(idx,new_line)
                                print(new_line)
                break
        idx = idx + 1
        end=len(all_lines)-1
    ###insert the z_lines
    idx=start
    print(z_lines)
    while idx <= end:
        line = all_lines[idx]
        if(line.strip() == "C     %s"%(qssa_spec_list[0].strip())):
            all_lines.insert(idx,"C\n")
            for l in z_lines:
                all_lines.insert(idx,l)
            break
        idx = idx + 1
    ###write lines
    f=open(out_file,"w")
    for l in all_lines:
        f.write(l)
    f.close()
    return
#****************************************************************
#****************************************************************
#****************************************************************
if __name__ == "__main__":
    #red_spec = find_sk_species("./test_add_NNH/red_new_qss_add_NNH.inp")
    #for i in range(len(red_spec)):
    #    print(i+1,red_spec[i])
    add_smh("NNH",red_file,out_file)
    add_ratt(new_file,sk_file,out_file,out_file)
    add_ratx(new_file,sk_file,out_file,out_file,qssa_flag)
    add_rdot(new_file,sk_file,out_file,out_file,qssa_flag)
    modify_getrates(new_file,sk_file,out_file,out_file,qssa_flag)
    if(not qssa_flag):
        add_ytcp(new_file,sk_file,out_file,out_file)
    if(qssa_flag):
        add_qssa(new_file,sk_file,out_file,out_file,qssa_spec)
