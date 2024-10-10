#import Surface_confined_inference as sci
import numpy as np
import os
loc="tmp_scan_results"
frequencies=[x+"_Hz" for x in ["3","9","15","21"]]
for i in range(0,2):
    curr_loc=os.path.join(loc, frequencies[i])  
    files=os.listdir(curr_loc)
    savearray=[]
    for j in range(0, len(files)):
        if j==0:
         savearray=np.loadtxt(os.path.join(curr_loc, files[j]))
        else:

         savearray=np.concatenate((savearray, np.loadtxt(os.path.join(curr_loc, files[j]))),axis=0)
       
    np.savetxt(os.path.join("scan_results", frequencies[i]+"_scan_results.txt"), np.array(savearray))   
