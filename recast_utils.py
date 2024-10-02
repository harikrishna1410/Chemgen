import sys
from add_species_and_reactions import *
import math
from scipy import sparse

extra_space = "         "
space = "  "
space_m1 = "     "

###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
###this function returns the maximum seperation in the C index in a window of nr reactions
def get_max_seperation_in_c(inp_file,specs,nr,\
                            rev=True,\
                            non_qssa=False,\
                            qssa_specs=None,\
                            ext="std",\
                            rpad_multiple=1):
    count,r_dict = get_r_dict(inp_file,\
                        rev=rev,\
                        non_qssa=non_qssa,\
                        qssa_specs=qssa_specs,\
                        ext=ext,\
                        rpad_multiple=rpad_multiple)
    rid_g = []
    max_len = -1
    rid_change = -1
    rn_list = list(r_dict.keys())
    old_rid = []
    for st in range(0,len(rn_list),nr):
        rid=[]
        for rnum in rn_list[st:st+nr]:
            reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"],remove_dups=False)
            reacts = list(reacts_dict.keys())
            prods = list(prods_dict.keys())
            rid = list(set(rid) | set(reacts))
        if(st > 0):
            rid_change = max(rid_change,len(rid) - len(list(set(rid) & set(old_rid))))
        old_rid = rid
        max_len = max(max_len,len(rid))
    return max_len,rid_change
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def add_new_array(dtype,name,size,lines,new_lines):
    new_lines.append("%s%s :: %s(%d)=(/&\n"%(space,dtype,name,size))
    new_lines += lines
    new_lines.append("%s/)\n"%(space))
    return new_lines
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
def append_new_str(new_str,curr_line,lines):
    if(len(curr_line)+len(new_str) >= 72):
        lines.append(curr_line+"&\n")
        curr_line =  "%s%s"%(space,new_str)
    else:
        curr_line = curr_line + new_str
    return curr_line,lines
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def filter_reversible_reactions(r_dict):
    rev_dict = {}
    irrev_dict = {}
    for rnum in r_dict.keys():
        if("=>" in r_dict[rnum]["eqn"] \
            and "<=>" not in r_dict[rnum]["eqn"]):
            irrev_dict[rnum] = r_dict[rnum]
        else:
            rev_dict[rnum] = r_dict[rnum]
    return rev_dict,irrev_dict
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
##this will give a reaction such that RF = RB = 1 so net rate rate 
##is always zero
def get_dummy_reaction(rtype="std"):
    rdict = {}
    rdict["eqn"] = ""
    rdict["arh"] = [1.0,0.0,0.0]
    if(rtype=="troe"):
        rdict["troe"] = {}
        rdict["troe"]["low"] = [1.0,0.0,0.0]
    ##
    return rdict
        
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def find_max_specs(inp_file):
    r_dict \
            = parse_reactions(inp_file,change_order=False)
    max_specs = 0
    max_reacts = 0
    max_prods = 0
    for rnum in r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"],remove_dups=False)
        max_specs = max(max_specs,len(list(set(list(reacts_dict.keys())+list(prods_dict.keys())))))
        max_reacts = max(max_reacts,len(list(reacts_dict.keys())))
        max_prods = max(max_prods,len(list(prods_dict.keys())))
    return max_specs,max_reacts,max_prods
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def find_max_third_body(inp_file):
    r_dict \
            = parse_reactions(inp_file,change_order=False)
    max_reacts = 0
    for rnum in r_dict.keys():
        if("third-body" in r_dict[rnum].keys()):
            reacts_dict=r_dict[rnum]["third-body"]
            max_reacts = max(max_reacts,len(list(reacts_dict.keys())))
    return max_reacts
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def filter_qssa_reactions(r_dict,qssa_specs):
    qssa_dict = {}
    non_qssa_dict = {}
    for rnum in r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
    
        my_qssa = list(set(qssa_specs) & set(reacts+prods))
        if(len(my_qssa)>0):
            qssa_dict[rnum] = r_dict[rnum]
        else:
            non_qssa_dict[rnum] = r_dict[rnum]
    return non_qssa_dict,qssa_dict 
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def filter_reactions(r_dict):
    ###filter troe
    std_dict = {}
    troe_dict = {}
    plog_dict = {}
    third_dict = {}
    
    for rnum in r_dict.keys():
        if("troe" in r_dict[rnum].keys()):
            troe_dict[rnum] =  r_dict[rnum]
        elif("PLOG" in r_dict[rnum].keys()):
            plog_dict[rnum] = r_dict[rnum]
        elif("third-body" in r_dict[rnum].keys()):
            third_dict[rnum] = r_dict[rnum]
        else:
            std_dict[rnum] = r_dict[rnum]

    return std_dict,troe_dict,plog_dict,third_dict
###******************************************************************************************* 
###******************************************************************************************* 
def get_arh_coef_lines(r_dict,troe=False):
    A_lines = []
    B_lines = []
    nr_calc = 0
    for rnum in r_dict.keys():
        ###add RF
        ##A
        if(troe):
            A = r_dict[rnum]["troe"]["low"][0]
            beta = r_dict[rnum]["troe"]["low"][1]
            Ea = r_dict[rnum]["troe"]["low"][2]
        else:
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
        ###log(A)
        aline = ("%s%21.15E,& \n"%(space,np.log(A)))\
                  .replace("E","D")
        #aline = aline+"&\n"
        ####Ea/R
        if(Ea>0):
            sign="-"
        else:
            sign="+"
        bline = bline + ("%s%21.15E,&\n"%(sign,np.abs(Ea)/R_c))\
                  .replace("E","D")
        ###
##        bline = bline+"&\n"
        
        A_lines.append(aline)
        B_lines.append(bline)
        nr_calc += 1
    A_lines[-1] = A_lines[-1].replace(",","")
    B_lines[-1] = B_lines[-1][:-3]+"&\n"
    return nr_calc,A_lines,B_lines
###******************************************************************************************* 
###******************************************************************************************* 
##pad=True: pads the coef with zero
def get_sk_map_coef(r_dict,sk_specs,pad=True,max_specs=None,remove_dups=False,sort=False):
    if(pad):
        if(max_specs is None):
            print("Error: I can't pad without max specs")
            sys.exit()
        if(max_specs%2 != 0):
            print("Error: max specs is not a multiple of 2")
            sys.exit()
#
    map_lines = []
    coef_lines = []
    if(not pad):
        idxs = []
        coef_sum_list = []

    map_line="%s"%(space)
    coef_line="%s"%(space)
    idx_line = "%s"%(space)
    coef_sum_line = "%s"%(space)
    nr_calc = 0
    start_idx = 0
    count = 0
    for rnum in r_dict.keys():
        coef_sum = 0
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"],remove_dups=remove_dups)
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        ##
        m_list = []
        c_list = []
        ###add reactants side
        for spec_name in reacts_dict.keys():
            spec_idx = sk_specs.index(spec_name)
            m_list.append(spec_idx)
            if(remove_dups):
                if(spec_name in prods_dict.keys()):
                    c_list.append(prods_dict[spec_name]-reacts_dict[spec_name])
                    del prods_dict[spec_name]
                else:
                    c_list.append(-reacts_dict[spec_name])
            else:
                c_list.append(-reacts_dict[spec_name])
            coef_sum = coef_sum + c_list[-1]
        ##
        if(pad):
            for i in range((max_specs//2)-len(reacts)):
                try:
                    m_list.append(m_list[-1])
                except:
                    m_list.append(0)
                c_list.append(0)
        ###add productss side
        for spec_name in prods_dict.keys():
            spec_idx = sk_specs.index(spec_name)
            m_list.append(spec_idx)
            c_list.append(prods_dict[spec_name])
            coef_sum = coef_sum + c_list[-1]
        if(sort):
            sorted_pairs = sorted(zip(m_list,c_list))
            m_list = [i[0] for i in sorted_pairs]
            c_list = [i[1] for i in sorted_pairs]
        ##
        c_len = len(c_list)
        if(pad):
            for i in range(max_specs-c_len):
                m_list.append(m_list[-1])
                c_list.append(0)
        else:
            idx_line,idxs = append_new_str("%d,"%start_idx,idx_line,idxs)
            start_idx += c_len
            coef_sum_line,coef_sum_list = append_new_str("%d,"%coef_sum,coef_sum_line,coef_sum_list)
        ##
        ##
        mstr = ",".join(["%d"%i for i in m_list])
        cstr = ",".join(["%d"%i for i in c_list])
        ##
        mstr += ","
        cstr += ","
        #
        map_line,map_lines = append_new_str(mstr,map_line,map_lines)
        coef_line,coef_lines = append_new_str(cstr,coef_line,coef_lines)
    ##
        nr_calc += 1
        count += len(c_list)

    coef_lines.append(coef_line[:-1]+"&\n") 
    map_lines.append(map_line[:-1]+"&\n") 
    if(not pad):
        idx_line,idxs = append_new_str("%d,"%start_idx,idx_line,idxs)
        idxs.append(idx_line[:-1]+"&\n")
    ##
    if(pad):
        return count,coef_lines,map_lines
    else:
        return count,coef_lines,map_lines,idxs,coef_sum_list
###******************************************************************************************* 
###******************************************************************************************* 
def get_rd_map_coef(r_dict,rd_specs,pad=True,max_specs=None,prods_flag=False,sort=False):
    if(pad):
        if(max_specs is None):
            print("Error: I can't pad without max specs")
            sys.exit()
    map_lines = []
    coef_lines = []
    if(not pad):
        idxs = []
        idx_line = "%s"%(space)
    map_line="%s"%(space)
    coef_line="%s"%(space)
    nr_calc = 0
    start_idx = 0
    count = 0
    for rnum in r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"],remove_dups=False)
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        ##
        m_list = []
        c_list = []
        if(prods_flag):
            spec_dict = prods_dict
        else:
            spec_dict = reacts_dict
        ##
        for spec_name in spec_dict.keys():
            if(spec_name in rd_specs):
                spec_idx = rd_specs.index(spec_name)
                m_list.append(spec_idx)
                c_list.append(spec_dict[spec_name])
        ##
        if(sort):
            sorted_pairs = sorted(zip(m_list,c_list))
            m_list = [i[0] for i in sorted_pairs]
            c_list = [i[1] for i in sorted_pairs]
        if(pad):
            for i in range(max_specs-len(c_list)):
                try:
                    m_list.append(m_list[-1])
                except:
                    m_list.append(0)
                c_list.append(0)
        ##
        mstr = ",".join(["%d"%i for i in m_list]) + ","
        cstr = ",".join(["%d"%i for i in c_list]) + ","
        #
        if(not pad):
            idx_line,idxs = append_new_str("%d,"%start_idx,idx_line,idxs)
            start_idx += len(c_list)
        #
        map_line,map_lines = append_new_str(mstr,map_line,map_lines)
        coef_line,coef_lines = append_new_str(cstr,coef_line,coef_lines)
        ##
        count += len(c_list)
        nr_calc += 1

    coef_lines.append(coef_line[:-1]+"&\n")
    map_lines.append(map_line[:-1]+"&\n")

    if(pad):
        return count,coef_lines,map_lines
    else:
        idx_line,idxs = append_new_str("%d,"%start_idx,idx_line,idxs)
        idxs.append(idx_line[:-1]+"&\n")
        return count,coef_lines,map_lines,idxs
###******************************************************************************************* 
        
def get_max_reactions_per_rblk(r_dict,specs,nr):
    nreacts = len(r_dict.keys())
    nspecs = len(specs)
    rn_list = list(r_dict.keys())
    map = np.zeros((nreacts,math.ceil(nreacts/nr)))

    for rid,rnum in enumerate(r_dict.keys()):
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
        ##
        ##
        for s in reacts+prods:
            if(s in specs):
                map[specs.index(s),rid//nr] += 1
    max_reacts = []
    for i in range(math.ceil(nreacts/nr)):
        max_reacts.append(np.amax(map[:,i]))

    return max_reacts


###******************************************************************************************* 
###******************************************************************************************* 
def get_coef_map_idx_wdot(r_dict,specs):
    ###
    map_lines = []
    coef_lines = []
    idxs = []

    map_line="%s"%(space)
    coef_line="%s"%(space)
    idx_line = "%s"%(space)
    start_idx = 0
    for spec_name in specs:
        nr_calc = 0
        m_list = []
        c_list = []
        for rnum in r_dict.keys():
            coef_sum = 0
            reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
            reacts = list(reacts_dict.keys())
            prods = list(prods_dict.keys())
            ##
            nr_calc += 1
            ##
            if(spec_name in prods_dict.keys() and spec_name in reacts_dict.keys()):
                c_list.append(prods_dict[spec_name]-reacts_dict[spec_name])
                m_list.append(nr_calc-1)
            elif(spec_name in reacts_dict.keys()):
                c_list.append(-reacts_dict[spec_name])
                m_list.append(nr_calc-1)
            elif(spec_name in prods_dict.keys()):
                c_list.append(prods_dict[spec_name])
                m_list.append(nr_calc-1)
            else:
                continue
            ##
        idx_line,idxs = append_new_str("%d,"%start_idx,idx_line,idxs)
        start_idx += len(c_list)
        ##
        mstr = ",".join(["%d"%i for i in m_list])
        cstr = ",".join(["%d"%i for i in c_list])
        #print(cstr)
        if(len(m_list) > 0):
            mstr += ","
            cstr += ","
        #
        map_line,map_lines = append_new_str(mstr,map_line,map_lines)
        coef_line,coef_lines = append_new_str(cstr,coef_line,coef_lines)
    ##
    coef_lines.append(coef_line[:-1]+"&\n")
    map_lines.append(map_line[:-1]+"&\n")
    idx_line,idxs = append_new_str("%d,"%start_idx,idx_line,idxs)
    idxs.append(idx_line[:-1]+"&\n")
    ##
    return start_idx,coef_lines,map_lines,idxs
###******************************************************************************************* 
###******************************************************************************************* 
def get_fcent_coef_lines(r_dict):
    lines = []
    for rnum in r_dict.keys():
        c = r_dict[rnum]["troe"]
        if("troe" not in c.keys()):
            l = [0.0,0.0,0.0,0.0,1.0,0.0]
        elif(len(c["troe"])==3):
            alpha = c["troe"][0]
            t1 = c["troe"][1]
            t2 = c["troe"][2]
            l = [1.0-alpha,t1,alpha,t2] + [0.0,1.0]
        else:
            alpha = c["troe"][0]
            t1 = c["troe"][1]
            t2 = c["troe"][2]
            t3 = c["troe"][3]
            l = [1.0-alpha,t1,alpha,t2,1.0,t3]
        for i in l:
            lines.append(("%s%21.15E,&\n"%(space,i)).replace("E","D"))
    lines[-1] = lines[-1].replace(",","")
    return lines
###******************************************************************************************* 
###******************************************************************************************* 
def get_third_body_coef(r_dict,rd_specs,pad=True,max_specs=None,sort=False):
    if(pad):
        if(max_specs is None):
            print("Error: I can't pad without max specs")
            sys.exit()
    map_lines = []
    coef_lines = []
    if(not pad):
        idxs = []
        idx_line = "%s"%(space)
    map_line="%s"%(space)
    coef_line="%s"%(space)
    nr_calc = 0
    start_idx = 0
    count = 0
    for rnum in r_dict.keys():
        if("third-body" in r_dict[rnum].keys()):
            spec_dict = r_dict[rnum]["third-body"]
        else:
            spec_dict = {}
        ##
        m_list = []
        c_list = []
        ##
        for spec_name in spec_dict.keys():
            if(spec_name in rd_specs):
                spec_idx = rd_specs.index(spec_name)
                m_list.append(spec_idx)
                c_list.append(spec_dict[spec_name])
        ##
        if(sort):
            sorted_pairs = sorted(zip(m_list,c_list))
            m_list = [i[0] for i in sorted_pairs]
            c_list = [i[1] for i in sorted_pairs]
        if(pad):
            for i in range(max_specs-len(c_list)):
                try:
                    m_list.append(m_list[-1])
                except:
                    m_list.append(0)
                c_list.append(0)
        ##
        mstr = ",".join(["%d"%i for i in m_list]) + ","
        cstr = ",".join(["%.2f"%i for i in c_list]) + ","
        #
        if(not pad):
            idx_line,idxs = append_new_str("%d,"%start_idx,idx_line,idxs)
            start_idx += len(c_list)
        #
        map_line,map_lines = append_new_str(mstr,map_line,map_lines)
        coef_line,coef_lines = append_new_str(cstr,coef_line,coef_lines)
        ##
        count += len(c_list)
        nr_calc += 1

    coef_lines.append(coef_line[:-1]+"&\n")
    map_lines.append(map_line[:-1]+"&\n")

    if(pad):
        return count,coef_lines,map_lines
    else:
        idx_line,idxs = append_new_str("%d,"%start_idx,idx_line,idxs)
        idxs.append(idx_line[:-1]+"&\n")
        return count,coef_lines,map_lines,idxs
###******************************************************************************************* 
###******************************************************************************************* 
##rpad_multiple is used when you have ext="all" where you merge are reactions into a single rdict.
##when doing this number of each type reactions is made multiple of the rpad_multiple. the dummy reactions
##are simply the empty reactions
####for now I am not adding PLOG
def get_r_dict(inp_file,rev=True,non_qssa=False,qssa_specs=None,ext="std",rpad_multiple=1):
    r_dict \
            = parse_reactions(inp_file,change_order=False)
    rnum=1
    rnum += len(r_dict.keys())
    std_dict,troe_dict,plog_dict,third_dict=filter_reactions(r_dict)
    if(ext != "all"):
        if(ext == "std"):
            r_dict = std_dict
        elif(ext == "troe"):
            r_dict = troe_dict
        elif(ext == "plog"):
            r_dict = plog_dict
        else:
            r_dict = third_dict
        ##
        if(non_qssa):
            r_dict,temp = filter_qssa_reactions(r_dict,qssa_specs)
        else:
            nq_dict,r_dict = filter_qssa_reactions(r_dict,qssa_specs)
            r_dict.update(nq_dict)
            
        if(rev):
            r_dict,std_irrev = filter_reversible_reactions(r_dict)
        nr = len(r_dict.keys())
        for i in range(math.ceil(nr/rpad_multiple)*rpad_multiple-nr):
            r_dict[rnum] = get_dummy_reaction(ext)
            rnum += 1
        count = [len(r_dict.keys())]
    else:
        count = []
        rlist = [std_dict,troe_dict,third_dict]
        rtype_list = ["std","troe","third"]
        r_dict = {}
        for d,rtype in zip(rlist,rtype_list):
            if(non_qssa):
                d,temp = filter_qssa_reactions(d,qssa_specs)
            else:
                temp,d = filter_qssa_reactions(d,qssa_specs)
                d.update(temp)
            if(rev):
                d,temp = filter_reversible_reactions(d)
            ###
            nr = len(d.keys())
            for i in range(math.ceil(nr/rpad_multiple)*rpad_multiple-nr):
                d[rnum] = get_dummy_reaction(rtype)
                rnum += 1
            r_dict.update(d)
            count.append(len(d.keys()))
    return count,r_dict
###******************************************************************************************* 
###******************************************************************************************* 
##this function returns the list of reactions with non-linear qssa for RF and RB
def get_non_linear_qssa(r_dict,qssa_specs):
    nl_qssa_rf = []
    nl_qssa_rb = []
    for rnum in r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
##
        my_qssa = list(set(qssa_specs) & set(reacts+prods))
        if(len(my_qssa)==0):
            continue
        
        my_qssa_r = list(set(qssa_specs) & set(reacts))
        my_qssa_p = list(set(qssa_specs) & set(prods))
        if(len(my_qssa_r)>1):
            nl_qssa_rf.append(rnum)
        if(len(my_qssa_p)>1):
            nl_qssa_rb.append(rnum)
    return nl_qssa_rf,nl_qssa_rb

###******************************************************************************************* 
def get_qssa_coef_dict(map_dict,pad=True,maxlen=2):
    c_dict = {}
    for rnum in map_dict.keys():
        m_list = map_dict[rnum]
        l = len(m_list)
        c_dict[rnum] = [1]*l
        if(pad):
            for i in range(maxlen-l):
                try:
                    map_dict[rnum].append(m_list[-1])
                except:
                    map_dict[rnum].append(0)
                c_dict[rnum].append(0)
    return map_dict,c_dict
###******************************************************************************************* 
###this function returns the non-zero elements of the A matrix of qssa in csr format.
##this is returned as a hash map where keys are the non-zero idx tuples and value is the 
##idx in the csr format. it also returns another dict with keys are rnums and values
##as the the list of tuples where this forward and backward reaction contributes to
def get_qssa_hash_map(r_dict,qssa_specs):
    A_mat = np.zeros((len(qssa_specs),len(qssa_specs)),dtype=bool)
    rf_a_dict = {}
    rb_a_dict = {}
    rf_b_dict = {}
    rb_b_dict = {}
###make diag 1
    for i in range(len(qssa_specs)):
        A_mat[i,i] = 1
    for rnum in r_dict.keys():
        reacts_dict,prods_dict=get_prod_reacts(r_dict[rnum]["eqn"])
        reacts = list(reacts_dict.keys())
        prods = list(prods_dict.keys())
##
        my_qssa = list(set(qssa_specs) & set(reacts+prods))
        rf_a_dict[rnum] = []
        rf_b_dict[rnum] = []
        rb_a_dict[rnum] = []
        rb_b_dict[rnum] = []
        if(len(my_qssa)==0):
            continue 
        my_qssa_r = list(set(qssa_specs) & set(reacts))
        my_qssa_p = list(set(qssa_specs) & set(prods))
        if(len(my_qssa_r) == 1 and len(my_qssa_p)==1):
            i=qssa_specs.index(my_qssa_r[0])
            j=qssa_specs.index(my_qssa_p[0])
            A_mat[i,j] = 1
            A_mat[j,i] = 1
            #
            rf_a_dict[rnum].append((i,i))
            rf_a_dict[rnum].append((j,i))
            rb_a_dict[rnum].append((j,j))
            rb_a_dict[rnum].append((i,j))

        elif(len(my_qssa_r)==1):
            i=qssa_specs.index(my_qssa_r[0])
            rf_a_dict[rnum].append((i,i))
            rb_b_dict[rnum].append(i)
        elif(len(my_qssa_p)==1):
            j=qssa_specs.index(my_qssa_p[0])
            rb_a_dict[rnum].append((j,j))
            rf_b_dict[rnum].append(j)
###get the csr matrix
    A_csr = sparse.csr_matrix(A_mat)
###convert the 2D idx to csr idx
    for rnum in r_dict.keys():
        for i,idx in enumerate(rf_a_dict[rnum]):
            rf_a_dict[rnum][i] = A_csr.indptr[idx[0]] + \
            A_csr.indices[A_csr.indptr[idx[0]]:A_csr.indptr[idx[0]+1]].tolist().index(idx[1])
        for i,idx in enumerate(rb_a_dict[rnum]):
            rb_a_dict[rnum][i] = A_csr.indptr[idx[0]] + \
            A_csr.indices[A_csr.indptr[idx[0]]:A_csr.indptr[idx[0]+1]].tolist().index(idx[1])
    return A_csr,rf_a_dict,rf_b_dict,rb_a_dict,rb_b_dict

###******************************************************************************************* 
###******************************************************************************************* 
def write_lines_from_dict_of_lists(ldict):
    ls = []
    l=""
    for k in ldict.keys():
        lstr = ",".join(["%d"%i for i in ldict[k]])
        l,ls = append_new_str(lstr+",",l,ls)
    ls.append(l[:-1]+"&\n")
    return ls
###******************************************************************************************* 
###******************************************************************************************* 
def get_qssa_map_coef(r_dict,qssa_specs):
    A_csr,rfa,rfb,rba,rbb = get_qssa_hash_map(r_dict,qssa_specs)
##
    start_idx = A_csr.indptr
    st_l = []
    l = ""
    for i in start_idx:
        l,st_l = append_new_str("%d,"%i,l,st_l)
    st_l.append(l[:-1]+"&\n")

    rfa,cfa = get_qssa_coef_dict(rfa)
    rfb,cfb = get_qssa_coef_dict(rfb,maxlen=1)
    rba,cba = get_qssa_coef_dict(rba)
    rbb,cbb = get_qssa_coef_dict(rbb,maxlen=1)

    rfa_ls = write_lines_from_dict_of_lists(rfa)
    rfb_ls = write_lines_from_dict_of_lists(rfb)
    rba_ls = write_lines_from_dict_of_lists(rba)
    rbb_ls = write_lines_from_dict_of_lists(rbb)
##
    cfa_ls = write_lines_from_dict_of_lists(cfa)
    cfb_ls = write_lines_from_dict_of_lists(cfb)
    cba_ls = write_lines_from_dict_of_lists(cba)
    cbb_ls = write_lines_from_dict_of_lists(cbb)
##
    return start_idx[-1],st_l,(rfa_ls,cfa_ls),(rfb_ls,cfb_ls),(rba_ls,cba_ls),(rbb_ls,cbb_ls)
###******************************************************************************************* 
###******************************************************************************************* 
