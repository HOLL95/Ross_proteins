import numpy as np
import matplotlib.pyplot as plt
import Surface_confined_inference as sci
import os
from pandas import DataFrame
loc="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Trumpet"
files=os.listdir(loc)
mvs=[]
averages=[]
colours=sci._utils.colours
fig, ax=plt.subplots(1,2)
mv_plots=[100e-3, 500e-3, 750e-3]
mv_counter=0
upper_val=-0.2
for file in files:
    data=np.loadtxt(os.path.join(loc, file), skiprows=1)
    splitfile=file.split("_")
    pgeloc=splitfile.index("PGE")
    mv=int(splitfile[pgeloc+1])/1e3
    
    if mv>50e-3 and mv<1000e-3:
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
            exclude=np.where((c_pot<-0.6) | (c_pot>upper_val))
            currents.append(c_curr[exclude]/mv)
            if mv in mv_plots:
                
                values=[-0.6, upper_val]
                idx1=np.where(c_pot<-0.6)
                idx2=np.where(c_pot>upper_val)
                
                
                ax[0].plot(c_pot[idx2], c_curr[idx2], color=colours[mv_counter])
                peakdx=np.where((c_pot>-0.6) & (c_pot<upper_val))
                ax[0].plot(c_pot[peakdx], c_curr[peakdx], linestyle="--", color="lightslategray")
                ax[0].plot()
                if i==1:
                    ax[0].plot(c_pot[idx1], c_curr[idx1], color=colours[mv_counter], label="%d mV s$^{-1}$" % (mv*1000))
                    mv_counter+=1
                    
                else:
                     ax[0].plot(c_pot[idx1], c_curr[idx1], color=colours[mv_counter])
                    

            
            
        averages.append(np.abs([np.mean(currents[0]), np.mean(currents[1])])/mv)
mvs=np.array(mvs)
ax[0].set_xlabel("Potential (V)")
ax[0].set_ylabel("Current (A)")
ax[0].legend()
averages=np.array(averages)
ax[1].scatter(mvs[:,0], averages[:,0], label="Forwards scan")
ax[1].scatter(mvs[:,1], averages[:,1], label="Backwards scan")
ax[1].set_xlabel("Scan rate (mV s$^{-1}$)")
ax[1].set_ylabel("Average non-Fardaic current magnitude (A s mV $^{-1}$)")
ax[1].legend()

plt.show()       

fig.savefig("DCVCDL.png", dpi=500)        
    
