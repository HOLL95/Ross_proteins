import numpy as np
import matplotlib.pyplot as plt
import Surface_confined_inference as sci
import os
from pandas import DataFrame
loc="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Trumpet"
files=os.listdir(loc)
mvs=[]
averages=[]
for file in files:
    data=np.loadtxt(os.path.join(loc, file), skiprows=1)
    splitfile=file.split("_")
    pgeloc=splitfile.index("PGE")
    mv=int(splitfile[pgeloc+1])/1e3
    
    if mv>50e-3 and mv<2000e-3:
        mvs.append([mv*1000, mv*1000])
        current=data[:,2]
        potential=data[:,1]
        idx_1=[10, len(current)//2]
        idx_2=[len(current)//2+10, -10]
        currents=[]
        for i in range(0, 2):
            if i==0:
                idx=idx_1
            else:
                idx=idx_2
            c_pot=potential[idx[0]:idx[1]]
            c_curr=current[idx[0]:idx[1]]
            exclude=np.where((c_pot<-0.6) | (c_pot>-0.3))
            currents.append(c_curr[exclude]/mv)
            plt.plot(c_pot[exclude], c_curr[exclude])
        plt.show()
        averages.append(np.abs([np.mean(currents[0]), np.mean(currents[1])])/mv)
mvs=np.array(mvs)
averages=np.array(averages)
plt.scatter(mvs[:,0], averages[:,0], label="Forwards scan")
plt.scatter(mvs[:,1], averages[:,1], label="Backwards scan")
plt.xlabel("Scan rate")
plt.ylabel("Average non-Fardaic current")
plt.legend()
plt.show()       
        
    
