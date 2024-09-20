import os
import numpy as np
import re
loc="/home/henryll/Downloads/Full_table.txt"
with open(loc, "r")as f:
    lines=f.readlines()
    for line in lines[1:]:
        linelist=re.split(r",\s+", line)
        numeric_line=[float(x) for x in linelist[1:-1]]
        
