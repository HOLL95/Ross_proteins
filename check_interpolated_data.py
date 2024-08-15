import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from scipy.optimize import fmin
abspath="/home/henryll/Documents/Experimental_data/Nat/m4D2/"
#abspath="/home/userfs/h/hll537/Documents/Experimental_data/m4D2"
dataloc="m4D2_Data"
dataloc="Interpolated"
Blankloc="Blank"
experiments=["FTACV", "PSV"]


abspath="/users/hll537/Experimental_data"
saved=abspath+"/Interpolated"
for i in range(1, len(experiments)):
    if experiments[i]=="FTACV":
        grouping=["mV.txt", "Hz"]
        rows=2
        cols=4
        
    elif experiments[i]=="PSV":
        grouping=["osc.txt", "Hz"]
        rows=2
        cols=2
    
    directory="/".join([abspath, dataloc, experiments[i]])
    
    files=os.listdir(directory)
    file_dict={}
    for j in range(0, len(files)):
        data=np.loadtxt("/".join([saved, experiments[i], files[j]]))
        time=data[:,0]
        current=data[:,1]
        potential=data[:,2]
        sci.plot.plot_harmonics(saved_time_series_data={"time":time, "current":current, "potential":potential, "harmonics":list(range(0, 10))}, xaxis="potential" )
        plt.show()

   
              
           

    
