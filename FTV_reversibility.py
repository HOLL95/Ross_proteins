import numpy as np
import matplotlib.pyplot as plt
import os
import re
import Surface_confined_inference as sci
loc="/home/henryll/Documents/Inference/M4D2/set9"
folders=os.listdir(loc)
hz_values=["3", "9", "15", "21"]
amp_values=["250"]
scatter_folders=[]
labels=[]
index_dict={}
for i in range(0, len(hz_values)):
    for j in range(0, len(amp_values)):
        desired_file="{0}Hz_FTV_Fourier_2_{1}".format(hz_values[i], amp_values[j])
        if desired_file in folders:
            scatter_folders.append(desired_file)
            labels.append("{0} Hz, {1} mV".format(hz_values[i], amp_values[j]))
            index_dict[desired_file]={"freq":hz_values[i], "amp":amp_values[j]}

data_loc="/home/henryll/Documents/Experimental_data/ForGDrive/Interpolated/FTACV"
harmonics=list(range(1, 10))
n_harms=len(harmonics)
fig, ax=plt.subplots()
best_harm=[9,5,4,2]


for i in range(0, len(best_harm)):
    print(i)
    pooled_loc=loc+"/"+scatter_folders[i]
    pooled_files=os.listdir(pooled_loc)
    pooled_name=[x for x in pooled_files if "Pooled" in x][0]
    table_loc="/".join([pooled_loc, pooled_name, "Full_table.txt"])
    full_data_loc=data_loc+"/"+index_dict[scatter_folders[i]]["amp"]
    data_file="FTACV_m4D2_PGE_59.60_mV_s-1_{0}_Hz_{1}_mV.txt".format(index_dict[scatter_folders[i]]["freq"],index_dict[scatter_folders[i]]["amp"])
    data=np.loadtxt(full_data_loc+"/"+data_file)
    time=data[:,0]
    current=data[:,1]
    potential=data[:,2]
    simulator=sci.LoadSingleExperiment("FTV_sims/FTV_simulator_{0}_Hz_{1}_mv.json".format(index_dict[scatter_folders[i]]["freq"],index_dict[scatter_folders[i]]["amp"]))
    
    get_optim_list=simulator._optim_list
    simulator.boundaries["E0"]=[-1, 1]
    e0=-0.42
    window=5
    simulator.fixed_parameters={"E0":e0, "phase":0, "cap_phase":0}
    v=simulator._internal_memory["input_parameters"]["v"]
    estart=simulator._internal_memory["input_parameters"]["E_start"]
    erev=simulator._internal_memory["input_parameters"]["E_reverse"]
    e0_time_1=(e0-estart)/v
    e0_time_2=((erev-estart)/v)+e0_time_1
    print(e0_time_2)
    farad_window=np.where(((time>e0_time_1-window) & (time<e0_time_1+window)) | ((time>e0_time_2-window) & (time<e0_time_2+window)))
    
    
    
    simulator.optim_list=get_optim_list[2:]
    with open(table_loc, "r") as f:
        lines=f.readlines()
        params=lines[0].split(",")
        Ru_idx=[x for x in range(0, len(params)) if "R_u" in params[x]][0]
        k0_idx=[x for x in range(0, len(params)) if "k_0" in params[x]][0]
        counter=0
        print(lines[0])
        for line in lines[1:2]:
            counter+=1
            get_all_numbers=re.finditer(r"(-)?(\d+\.)?\d+(e(\+|-)\d+)?(?=,)",line)
            numbers=[float(x.group()) for x in get_all_numbers]
            
            
            k0_val=numbers[k0_idx]
            labels=["Best", "Reversible"]
            colours=["red","black"]
            linestyles=["-", "--"]
            step=20
            kvals=np.arange(1, 1500, step)
            repeats=1
            diffs=np.zeros( len(kvals)-1)
            for p in range(0, repeats):
                for m in range(0, len(kvals)):
                    numbers[k0_idx]=kvals[m]
                    params=numbers[3:-1]
                    params[0]=kvals[m]
                    #
                    params[2]=1000

                    
                    print(params)
                    #print(simulator.optim_list)
                    simcurrent=simulator.dim_i(simulator.Dimensionalsimulate(params, time))*1e6
                    zeroed_current=np.zeros(len(time))
                    zeroed_current[farad_window]=simcurrent[farad_window]
                    #plt.plot(time, simcurrent)
                    #plt.plot(time, zeroed_current)
                    #plt.show()
                    #newharms=np.abs(sci.plot.generate_harmonics(time, simcurrent, one_sided=True, hanning=True, func=None,harmonics=[best_harm[i]-1, best_harm[i]]))
                    #sci.plot.plot_harmonics(data_data={"time":time, "current":simcurrent,"harmonics":[4,5]}, plot_func=np.abs, hanning=True)
                    #plt.plot(time, newharms[-1,:])
                    #plt.show()
                    
                    if m>0:
                        diffs[m-1]=sci._utils.RMSE(zeroed_current, oldcurrent)
                        
                        oldcurrent=zeroed_current
                        
   
                    else:
                        oldcurrent=zeroed_current
    

            ax.plot(kvals[:-1], diffs, label="{0} Hz".format(index_dict[scatter_folders[i]]["freq"]))    #np.mean(diffs, axis=0)
            #ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("$k^0$ $s^{-1}$")
            ax.set_ylabel("Mean error with $\\uparrow$"+str(step)+ " $s^{-1}$ ($\\mu$A)")
            
plt.legend()
plt.show()