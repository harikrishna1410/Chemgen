###this script reads the chemkin files and list of qssa species to recast the 
### unlike the recast_getreates.py this is written to fit the structure of getrates_gpu4.f 

from add_species_and_reactions import *
from recast_utils import *
import sys


rocsparse=False

    
###******************************************************************************************* 
###******************************************************************************************* 
###writes the function to calculate the forward reactions
def write_rev_kf_coef(inp_file,non_qssa=False,qssa_specs=None,ext="std",rpad_multiple = 1):
    if(non_qssa):
        if(qssa_specs is None):
            print("Error:can't filter qssa reactions without qssa_specs")
            sys.exit()
    ##
    count,r_dict = get_r_dict(inp_file,rev=True,non_qssa=non_qssa,
                qssa_specs=qssa_specs,ext=ext,rpad_multiple = rpad_multiple)
    if(ext=="all"):
        print("nreacts:",count)
    print("nreacts %s:%d"%(ext,len(r_dict.keys())))
    ##
    new_lines = []
    if(ext == "std"):
        nr_calc,A_lines,B_lines = get_arh_coef_lines(r_dict)
        new_lines = add_new_array("real","A_std",nr_calc,A_lines,new_lines)
        new_lines = add_new_array("real","B_std",2*nr_calc,B_lines,new_lines)
    elif(ext == "troe"):
        nr_calc,A_lines,B_lines = get_arh_coef_lines(r_dict)
        new_lines = add_new_array("real","A_inf_troe",nr_calc,A_lines,new_lines)
        new_lines = add_new_array("real","B_inf_troe",2*nr_calc,B_lines,new_lines)
        ##
        nr_calc,A_lines,B_lines = get_arh_coef_lines(r_dict,troe=True)
        new_lines = add_new_array("real","A_0_troe",nr_calc,A_lines,new_lines)
        new_lines = add_new_array("real","B_0_troe",2*nr_calc,B_lines,new_lines)
    elif(ext == "plog"):
        nr_calc,A_lines,B_lines = get_arh_coef_lines(r_dict)
        new_lines = add_new_array("real","A_plog",nr_calc,A_lines,new_lines)
        new_lines = add_new_array("real","B_plog",2*nr_calc,B_lines,new_lines)
    elif(ext == "third"):
        nr_calc,A_lines,B_lines = get_arh_coef_lines(r_dict)
        new_lines = add_new_array("real","A_third",nr_calc,A_lines,new_lines)
        new_lines = add_new_array("real","B_third",2*nr_calc,B_lines,new_lines)
    else:
        nr_calc,A_lines,B_lines = get_arh_coef_lines(r_dict)
        new_lines = add_new_array("real","A_all",nr_calc,A_lines,new_lines)
        new_lines = add_new_array("real","B_all",2*nr_calc,B_lines,new_lines)
        ##
        count,r_dict = get_r_dict(inp_file,rev=True,non_qssa=non_qssa,\
                qssa_specs=qssa_specs,ext="troe",rpad_multiple = rpad_multiple)
        nr_calc,A_lines,B_lines = get_arh_coef_lines(r_dict,troe=True)
        new_lines = add_new_array("real","A_0_troe",nr_calc,A_lines,new_lines)
        new_lines = add_new_array("real","B_0_troe",2*nr_calc,B_lines,new_lines)
    return new_lines
###******************************************************************************************* 
###******************************************************************************************* 
def write_rev_eg_coef(inp_file,non_qssa=False,qssa_specs=None,ext="std",rpad_multiple=1):
    if(non_qssa):
        if(qssa_specs is None):
            print("Error:can't filter qssa reactions without qssa_specs")
            sys.exit()
    ###find the skeletal species
    specs = find_sk_species(inp_file)
    ##find max specs
    max_specs,max_reacts,max_prods = find_max_specs(inp_file)
    ##get the reactions dictionary
    count,r_dict = get_r_dict(inp_file,rev=True,non_qssa=non_qssa,\
            qssa_specs=qssa_specs,ext=ext,rpad_multiple=rpad_multiple)
    ##generate coefs
    count,coef_lines,map_lines = \
    get_sk_map_coef(r_dict,specs,pad=True,max_specs=2*max_reacts,remove_dups=False,sort=False)
    ##
    ##write coefs
    nr_calc = len(list(r_dict.keys()))
    print("nelement in sk map:",count)
    ##
    new_lines = []
    new_lines = add_new_array("integer","sk_map_%s"%(ext),count,map_lines,new_lines)
    new_lines = add_new_array("integer","sk_coef_%s"%(ext),count,coef_lines,new_lines)
    ##
    return new_lines
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def write_rev_wdot_coef(inp_file,rd_specs,non_qssa=False,qssa_specs=None,ext="std"):
    if(non_qssa):
        if(qssa_specs is None):
            print("Error:can't filter qssa reactions without qssa_specs")
            sys.exit()
    ##get the reactions dictionary
    count,r_dict = get_r_dict(inp_file,rev=True,non_qssa=non_qssa,qssa_specs=qssa_specs,ext=ext)

    count,coef_lines,map_lines,idxs=get_coef_map_idx_wdot(r_dict,rd_specs)
    new_lines = []
    new_lines = add_new_array("integer","wdot_map_%s"%(ext),count,map_lines,new_lines)
    new_lines = add_new_array("integer","wdot_coef_%s"%(ext),count,coef_lines,new_lines)
    new_lines = add_new_array("integer","wdot_start_idx_%s"%(ext),len(rd_specs)+1,idxs,new_lines)
    ##
    return new_lines
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def write_rev_net_rate_lines(inp_file,rd_specs,non_qssa=False,qssa_specs=None,ext="std",pad=True,rpad_multiple=1):
    if(non_qssa):
        if(qssa_specs is None):
            print("Error:can't filter qssa reactions without qssa_specs")
            sys.exit()
    ##
    max_specs,max_reacts,max_prods = find_max_specs(inp_file)
    count,r_dict = get_r_dict(inp_file,rev=True,non_qssa=non_qssa,qssa_specs=qssa_specs
                            ,ext=ext,rpad_multiple=rpad_multiple)
    nr = len(r_dict.keys())
    if(pad) :
        count,coef_lines,map_lines=\
            get_rd_map_coef(r_dict,rd_specs,pad=pad,max_specs=max_reacts,prods_flag=False,sort=False)
    else:
        count,coef_lines,map_lines,idx_lines=\
            get_rd_map_coef(r_dict,rd_specs,pad=pad,max_specs=max_reacts,prods_flag=False,sort=False)
    ##
    new_lines = []
    new_lines = add_new_array("integer","map_r_%s"%ext,count,map_lines,new_lines)
    new_lines = add_new_array("integer","coef_r_%s"%ext,count,coef_lines,new_lines)
    if(not pad):
        new_lines = add_new_array("integer","start_idx_r_%s"%ext,nr+1,idx_lines,new_lines)
    print("nelem in rmap:",count)
    ##******************products*****************************************
    if(pad):
        count,coef_lines,map_lines=\
        get_rd_map_coef(r_dict,rd_specs,pad=pad,max_specs=max_reacts,prods_flag=True,sort=False)
    else:
        count,coef_lines,map_lines,idx_lines=\
        get_rd_map_coef(r_dict,rd_specs,pad=pad,max_specs=max_reacts,prods_flag=True,sort=False)
    ##
    new_lines = add_new_array("integer","map_p_%s"%ext,count,map_lines,new_lines)
    new_lines = add_new_array("integer","coef_p_%s"%ext,count,coef_lines,new_lines)
    if(not pad):
        new_lines = add_new_array("integer","start_idx_p_%s"%ext,nr+1,idx_lines,new_lines)
    print("nelem in pmap:",count)
    
    return new_lines
###******************************************************************************************* 
###******************************************************************************************* 
###writes the function to calculate the forward reactions
def write_troe_rev_fcent_coef(inp_file,non_qssa=False,qssa_specs=None,rpad_multiple=1):
    if(non_qssa):
        if(qssa_specs is None):
            print("Error:can't filter qssa reactions without qssa_specs")
            sys.exit()
    print("writing forward reaction rate")
    ###
    count,r_dict = get_r_dict(inp_file,rev=True,non_qssa=non_qssa,\
                    qssa_specs=qssa_specs,ext="troe",rpad_multiple=rpad_multiple)
    lines = get_fcent_coef_lines(r_dict)
    ##
    new_lines = []
    new_lines = add_new_array("real","fcent_coef",len(r_dict.keys())*6,lines,new_lines)
    ##
    return new_lines
###******************************************************************************************* 
###******************************************************************************************* 
###******************************************************************************************* 
def write_rev_third_lines(inp_file,rd_specs,non_qssa=False,qssa_specs=None,ext="troe",rpad_multiple=1):
    if(non_qssa):
        if(qssa_specs is None):
            print("Error:can't filter qssa reactions without qssa_specs")
            sys.exit()
    if(not (ext == "troe" or ext =="third")):
        print("ERROR:",ext," is not troe or third")
        sys.exit()
    ###find the skeletal species
    max_specs = find_max_third_body(inp_file)
    print("max third",max_specs)
    count,r_dict = get_r_dict(inp_file,rev=True,non_qssa=non_qssa,\
                        qssa_specs=qssa_specs,ext=ext,rpad_multiple=rpad_multiple)
    ## 
    sk_specs = find_sk_species(inp_file)
    count,coef_lines,map_lines=\
        get_third_body_coef(r_dict,rd_specs,pad=True,max_specs=max_specs,sort=False)
    ##
    nr_calc = len(list(r_dict.keys()))
    new_lines = []
    new_lines = add_new_array("integer","third_map_%s"%ext,count,map_lines,new_lines)
    new_lines = add_new_array("real","third_eff_%s"%ext,count,coef_lines,new_lines)
    print("nelem in third:",count)
    
    return new_lines
##*************************************************************
def write_qssa_coef(inp_file,qssa_specs,ext="std",rpad_multiple=1):
    ##get the reactions dictionary
    count_l,r_dict = get_r_dict(inp_file,rev=True,non_qssa=False,\
            qssa_specs=qssa_specs,ext=ext,rpad_multiple=rpad_multiple)
    nnz_A,start_idx_A,rfa,rfb,rba,rbb =  get_qssa_map_coef(r_dict,qssa_specs)
    count = sum(count_l)
    c=0
    for i in rfb[0]:
        c+=i.count(",")
    print(c+1)
    c=0
    for i in rfa[0]:
        c+=i.count(",")
    print(c+1)
    ##
    new_lines = []
    print("NNZ_A QSSA %s:%d"%(ext,nnz_A))
    new_lines = add_new_array("integer","start_idx_qssa_%s"%(ext),len(qssa_specs)+1,start_idx_A,new_lines)
    new_lines = add_new_array("integer","rfa_%s"%(ext),count*2,rfa[0],new_lines)
    new_lines = add_new_array("integer","cfa_%s"%(ext),count*2,rfa[1],new_lines)
    new_lines = add_new_array("integer","rfb_%s"%(ext),count,rfb[0],new_lines)
    new_lines = add_new_array("integer","cfb_%s"%(ext),count,rfb[1],new_lines)
    new_lines = add_new_array("integer","rba_%s"%(ext),count*2,rba[0],new_lines)
    new_lines = add_new_array("integer","cba_%s"%(ext),count*2,rba[1],new_lines)
    new_lines = add_new_array("integer","rbb_%s"%(ext),count,rfb[0],new_lines)
    new_lines = add_new_array("integer","cbb_%s"%(ext),count,rfb[1],new_lines)
    ##
    return new_lines
#**************************************************************************************
#**************************************************************************************
#**************************************************************************************
def write_cmap_qssa(inp_file,qssa_specs):
    ##
    sk_specs = find_sk_species(inp_file)
    lines = []
    lines.append(",".join(["%d"%(sk_specs.index(s)) for s in qssa_specs])+"&\n")
    new_lines = []
    new_lines = add_new_array("integer","c_map_qssa",len(qssa_specs),lines,new_lines)
    ##
    return new_lines

###******************************************************************************************* 
###******************************************************************************************* 
if __name__ == "__main__":
    ###outfile
    if(not rocsparse):
        out_file = os.path.join("./test_gpu","coef_m.f90")
    else:
        out_file = os.path.join("./test_gpu","coef_m_rocsparse.f90")
    ###underlying skeletal mechanism
    inp_file = os.path.join("./test_gpu","chem.ske50.inp")
    ##therm.dat file
    therm_file=os.path.join("./test_gpu","therm.dat")
    ##
    qssa_file = os.path.join("./test_gpu","qssa_spec.txt")
    non_qssa_file = os.path.join("./test_gpu","non_qssa_spec.txt")
    ##
    f=open(qssa_file,"r")
    qssa_specs = [i.strip() for i in f.readlines()]
    f.close()
    ##
    f=open(non_qssa_file,"r")
    non_qssa_specs = [i.strip() for i in f.readlines()]
    f.close()
    specs = find_sk_species(inp_file)
    maps = []
    for s in non_qssa_specs:
        maps.append(specs.index(s))
    f=open("./test_gpu/map.txt","w")
    f.write("%s\n"%(",".join(["%d"%i for i in maps])))
    f.close()
#    write_getrates(inp_file,out_file)
#    write_forward_rate_const_func(inp_file,out_file)
    non_qssa = False
    merge = True
    rpad_multiple = 1
    line = "!************************************************************************\n"
    coef_lines = []
    if(not merge):
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_kf_coef(inp_file,non_qssa,qssa_specs,ext="std")
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_eg_coef(inp_file,non_qssa,qssa_specs,ext="std")
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_net_rate_lines(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="std",pad=False)
        coef_lines += ["\n",line,"\n"]
        if(rocsparse):
            coef_lines += write_std_rev_wdot_coef(inp_file,non_qssa_specs,True,qssa_specs,ext="std")
        coef_lines += ["\n",line,"\n"]
        ## troe
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_kf_coef(inp_file,non_qssa,qssa_specs,ext="troe")
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_eg_coef(inp_file,non_qssa,qssa_specs,ext="troe")
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_net_rate_lines(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="troe")
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_troe_rev_fcent_coef(inp_file,non_qssa,qssa_specs)
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_third_lines(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="troe")
        coef_lines += ["\n",line,"\n"]
        ###third-body
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_kf_coef(inp_file,non_qssa,qssa_specs,ext="third")
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_eg_coef(inp_file,non_qssa,qssa_specs,ext="third")
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_net_rate_lines(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="third")
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_third_lines(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="third")
        coef_lines += ["\n",line,"\n"]
    else:
        sk_specs = find_sk_species(inp_file)
        max_diff,ch = get_max_seperation_in_c(inp_file,sk_specs,4,\
                            rev=True,\
                            non_qssa=False,\
                            qssa_specs=qssa_specs,\
                            ext="all",\
                            rpad_multiple=1)
        print("max diff",max_diff,ch)
        count_nq,r_dict = get_r_dict(inp_file,rev=True,non_qssa=True,
                qssa_specs=qssa_specs,ext="all",rpad_multiple = 1)
        count_all,r_dict = get_r_dict(inp_file,rev=True,non_qssa=False,
                qssa_specs=qssa_specs,ext="all",rpad_multiple = 1)
        max_reacts = get_max_reactions_per_rblk(r_dict,qssa_specs,8)
        print("max reacts per blk:",max_reacts)
        print("****************mech info***************************")
        print("non-qssa count:",count_nq)
        print("all count:",count_all)
        print("****************mech info***************************")
        ## troe
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_kf_coef(inp_file,non_qssa,qssa_specs,ext="all",rpad_multiple=rpad_multiple)
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_eg_coef(inp_file,non_qssa,qssa_specs,ext="all",rpad_multiple=rpad_multiple)
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_net_rate_lines(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="all",rpad_multiple=rpad_multiple)
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_troe_rev_fcent_coef(inp_file,non_qssa,qssa_specs,rpad_multiple=rpad_multiple)
        coef_lines += ["\n",line,"\n"]
        ##coef_lines += write_rev_wdot_coef(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="all")
        coef_lines += write_rev_third_lines(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="troe",rpad_multiple=rpad_multiple)
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_rev_third_lines(inp_file,non_qssa_specs,non_qssa,qssa_specs,ext="third",rpad_multiple=rpad_multiple)
        coef_lines += ["\n",line,"\n"]
        coef_lines += write_qssa_coef(inp_file,qssa_specs,ext="all",rpad_multiple=rpad_multiple)
        coef_lines += write_cmap_qssa(inp_file,qssa_specs)

    #count,r_dict = get_r_dict(inp_file,True,True,qssa_specs=qssa_specs,ext="troe")
    #print(r_dict.keys())
    #count,r_dict = get_r_dict(inp_file,True,True,qssa_specs=qssa_specs,ext="third")
    #print(r_dict.keys())
    #count,r_dict = get_r_dict(inp_file,True,False,qssa_specs=qssa_specs,ext="all")
    #temp,r_dict = filter_qssa_reactions(r_dict,qssa_specs)
    #A_csr,rfa,rfb,rba,rbb = get_qssa_hash_map(r_dict,qssa_specs)
    #rfa,ca = get_qssa_coef_dict(rfa)
    #print("nnz:",A_csr.indptr[-1])
    #for rnum in rfa.keys():
    #    print(rfa[rnum],ca[rnum])
    f= open(out_file,"w")
    f.write("  module coef_m\n")
    f.write("  implicit none\n")
    for l in coef_lines:
        f.write(l)
    f.write("  end module coef_m\n")
    f.close()
