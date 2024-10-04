space = " "
"""
this function adds a fortran array declaration to the file
"""
def add_new_array(dtype,name,size,lines,new_lines):
    new_lines.append("%s%s :: %s(%d)=(/&\n"%(space,dtype,name,size))
    new_lines += lines
    new_lines.append("%s/)\n"%(space))
    return new_lines

"""
this function adds the trailing end
"""
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


#this function calculates the number of species per thread block
#it is always a power of 2
def calculate_possible_nsp_per_block(chem,nreact_per_block,veclen,threads_per_block_ulimit=256):
    import math
    max_sp = chem.find_max_specs(good_number=True)
    ##max_sp is the maximum number of species per thread block so that nsp_per_thread > 0
    ##we also need to make sure that nsp_per_block*nreact_per_block*veclen <= threads_per_block_ulimit
    nsp_per_block = [2**i for i in range(0,int(math.log2(max_sp))+1) 
                     if 2**i*nreact_per_block*veclen <= threads_per_block_ulimit]
    return nsp_per_block