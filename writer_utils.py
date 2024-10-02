


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
