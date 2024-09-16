import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from scipy.optimize import fmin
from scipy.signal import decimate
abspath="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Raw"
#abspath="/home/userfs/h/hll537/Documents/Experimental_data/m4D2"
dataloc="m4D2_Data"
Blankloc="Blank"
experiments=["FTACV", "PSV"]

saved="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Interpolated"
saved2="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Unprocessed"

files=os.listdir(abspath)
file_dict={}
for j in range(0, len(files)):
    try:
        data=np.loadtxt(abspath+"/"+files[j], delimiter=",", skiprows=1)
    except:
         data=np.loadtxt(abspath+"/"+files[j])
    time=decimate(data[:,2], 3)
    potential=decimate(data[:,5], 3)
    current=decimate(data[:,4],3)
    if "PSV" in files[j]:
        loc=["PSV"]
        
    else:
        continue
        loc=["FTACV"]
        if "280_mV" in files[j]:
            loc +=["280"]
        else:
            loc +=["80"]
   
    if loc[0]=="FTACV":
        chopped_time=time
    else:
        freq=sci.get_frequency(time, current)
        end_time=35/freq
        start_time=0
        chopped_time=time[np.where((time<end_time) & (time>=start_time))]
        
    interped_time=np.linspace(chopped_time[0], chopped_time[-1], len(chopped_time))
    interped_potential=np.interp(interped_time, time, potential)
    interped_current=np.interp(interped_time, time, current)
    #if experiments[i]=="PSV":
    #    plt.plot(interped_potential, interped_current)
    #    plt.show()
    #plt.plot(time, potential)
    #plt.scatter(range(0, len(time)), time)
    #plt.show()
    new_file_idx=files[j].index(".csv")
    new_file=files[j][:new_file_idx]+".txt"
    np.savetxt("/".join([saved]+loc+[new_file]), np.column_stack((interped_time, interped_current, interped_potential)))
    #np.savetxt("/".join([saved2]+loc+[new_file]), np.column_stack((time, current, potential)))
   
              
           

    
