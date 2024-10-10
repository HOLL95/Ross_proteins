import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
loc="Profile_likelihoods"
freqs=["3_Hz", "9_Hz","15_Hz"]
files=os.listdir(loc)
for i in range(1, len(freqs)):
    file=[x for x in files if freqs[i] in x][0]
    data=np.loadtxt(os.path.join(loc,file))
    
    r_vals=sorted(np.unique(data[:,0]))
    datadict={key:{} for key in r_vals}
    for j in range(0, len(data[:,0])):
        datadict[data[j,0]][data[j,1]]=data[j,2]
    k_vals=sorted(datadict[r_vals[0]].keys())
    results_array=np.zeros((len(r_vals), len(k_vals)))
    k_vals_plot=[]
    for k in range(0, len(r_vals)):
        scores=[datadict[r_vals[k]][x] for x in k_vals]
        
        if r_vals[k]>1000:
            plt.semilogx(k_vals, scores, label=round(r_vals[k],2))
            plt.legend()    
            plt.show()
            
   #7.649843288033999897e+02
   #2.564428524099999777e+03