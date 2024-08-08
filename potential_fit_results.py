import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from scipy.optimize import fmin
abspath="/home/henryll/Documents/Experimental_data/Nat/m4D2/"
#abspath="/home/userfs/h/hll537/Documents/Experimental_data/m4D2"
dataloc="m4D2_Data"
Blankloc="Blank"
experiments=["FTACV", "PSV"]


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
        file_dict[files[j]]=np.loadtxt(directory+"/"+files[j])
    file_list=list(file_dict.keys())

    for j in range(0, 1):
        sortdict={}
        for m in range(0, len(file_list)):
            split=file_list[m].split("_")  
            
            idx=split.index(grouping[j])
            colkey=split[idx-1]
            plotkey=split[split.index(grouping[j-1])-1]

            if colkey in sortdict.keys():
                sortdict[colkey][plotkey]=file_dict[file_list[m]]
            else:
                sortdict[colkey]={plotkey:file_dict[file_list[m]]}
        
        plot_keys=list(sortdict.keys())
        plot_keys=[str(x) for x in sorted([int(x) for x in plot_keys])]
        print(plot_keys)
        for m in range(0, len(plot_keys)):

            pk=plot_keys[m]
            trace_keys=[str(x) for x in sorted([int(x) for x in sortdict[pk]])]
            print(trace_keys)
            for q in range(0, len(trace_keys)): 
                fig, axis=plt.subplots()

                tk=trace_keys[q]
                if experiments[i]=="FTACV":
                
                
                    time=sortdict[pk][tk][:,0]
                    current=sortdict[pk][tk][:,1]*1e6
                    potential=sortdict[pk][tk][:,2]
                elif experiments[i]=="PSV":
                    time=sortdict[pk][tk][:,0]
                    current=sortdict[pk][tk][:,1]*1e6
                    potential=sortdict[pk][tk][:,2]
                    freq=sci.get_frequency(time, current)
                    get_rid=5/freq
                    idx=np.where(time>get_rid)
                    time=time[idx]
                    potential=potential[idx]
                    current=current[idx]
                if experiments[i]=="FTACV":
                    rowdx=m
                    lab=" mV data"
                elif experiments[i]=="PSV":
                    rowdx=q//cols
                    lab=" oscillations data"
                #axis=ax[rowdx, q%cols]
                #p_est, p_inf, sim_est, sim_inf=sci.infer.get_input_parameters(time, potential, current,experiments[i], optimise=True, return_sim_values=True)
                p_est=sci.infer.get_input_parameters(time, potential, current,experiments[i], optimise=False)
                amp=p_est["Edc"]
                def var_phase_sine(params, return_error=True):
                    
                    
                    omega=params[0]
                    phase=params[1]
                    phase_omega=params[2]
                    phase_phase=params[3]
                    phase_amp=params[4]
                    edc=params[5]
                    amp=params[6]
                    interior_phase=phase_amp*np.sin(phase_omega*time+phase_phase)
                    value=2*edc+(amp*np.sin((2*np.pi*omega*time)+phase+interior_phase))
                    plain_sin=2*edc+(amp*np.sin((2*np.pi*omega*time)+phase))
                    
                    """
                    fig, ax=plt.subplots(2,2)
                    ax[0,0].plot(time, potential)
                    ax[0,0].plot(time, value)
                    
                    ax[1,0].plot(value-potential)
                    ax[0,1].plot(time, potential)
                    ax[0,1].plot(time, plain_sin)
                    
                    ax[1,1].plot(potential-plain_sin)
                    plt.show()"""
                    if return_error==True:
                        return sci._utils.RMSE(value, potential)
                    else:
                        return value
                
                minimisation=fmin(var_phase_sine,[p_est["omega"], p_est["phase"], 4, 0,1,p_est["Edc"],amp ])
                print(list(minimisation))
                axis.plot(time, var_phase_sine(minimisation, return_error=False))
                
                #axis.plot(time, potential-sim_inf, alpha=0.5, color=sci._utils.colours[2], label="residual")
                axis.set_title(tk+" Hz")
                axis.plot(time, potential, label=pk+lab)
                axis.plot(time, potential-var_phase_sine(minimisation, return_error=False)+p_est["Edc"])
                axis.set_xlabel("Time (s)")
                if q==0:
                    axis.set_ylabel("Potential (V)")
                #axis.plot(time, sim_inf, linestyle="--", label="fitted")
                plt.show()
        """ax[0,0].legend()
        if experiments[i]=="FTACV":
            ax[1,0].legend()"""
        plt.show()

           

    
