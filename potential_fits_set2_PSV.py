import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from scipy.optimize import fmin
abspath="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Interpolated"
#abspath="/home/userfs/h/hll537/Documents/Experimental_data/m4D2"

Blankloc="Blank"
experiments=["FTACV", "PSV"]
freqs=["9_Hz", "15_Hz"]

for i in range(1, 2):
    if experiments[i]=="FTACV":
        grouping=["mV.txt", "Hz"]
        additional_dirs=["280"]
        plot_keys=additional_dirs
    elif experiments[i]=="PSV":
        grouping=["osc.txt", "Hz"]
        additional_dirs=[""]
        plot_keys=["100_osc"]
    for j in range(0, len(additional_dirs)):

        directory="/".join([abspath, experiments[i], additional_dirs[j]])
        files=os.listdir(directory)

      
        
        for m in range(0, len(files)):

            tk=plot_keys[j]
            pk=[x for x in freqs if x in files[m]]
            try:
                pk =pk[0]
            except:
                continue
            data=np.loadtxt(directory+"/"+files[m])
            
            time=data[:,0]
            
            
            current=data[:,1]
            potential=data[:,2]
            freq=sci.get_frequency(time, current)
            first_idx=np.where(time>2/freq)
            time=time[first_idx]
            current=current[first_idx]
            potential=potential[first_idx]

            
                
            p_est, p_inf, sim_est, sim_inf=sci.infer.get_input_parameters(time, potential, current,experiments[i], 
                                                                            optimise=True, 
                                                                            return_sim_values=True, 
                                                                            sinusoidal_phase=False,
                                                                            sigma=0.075,
                                                                            runs=5
                                                                            )
            #p_est=sci.infer.get_input_parameters(time, potential, current,experiments[i], optimise=False)
            print("experiments_dict['{0}']['{1}']['{2}']=".format(experiments[i], pk, tk), p_inf)
            print(p_inf)
            init=p_inf
            init["Surface_coverage"]=1e-10
            init["Temp"]=298
            init["N_elec"]=1
            init["area"]=0.07
            fig,axis=plt.subplots()
            twinx=axis.twinx()
            axis.plot(potential)
            axis.plot(sim_inf)
            twinx.plot(potential-sim_inf)
            plt.show()
            
            """axis.plot(time, potential-sim_inf, alpha=0.5, color=sci._utils.colours[2], label="residual")
            axis.set_title(tk+" Hz")
            axis.plot(time, potential, label=pk+lab)
            axis.set_xlabel("Time (s)")
            if q==0:
                axis.set_ylabel("Potential (V)")
            axis.plot(time, sim_inf, linestyle="--", label="fitted")"""
                


           

    
