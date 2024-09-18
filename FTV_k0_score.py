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


for i in range(0, 3):
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
    FT=simulator.experiment_top_hat(time, current)
    with open(table_loc, "r") as f:
        lines=f.readlines()
        params=lines[0].split(",")
        Ru_idx=[x for x in range(0, len(params)) if "R_u" in params[x]][0]
        k0_idx=[x for x in range(0, len(params)) if "k_0" in params[x]][0]
        counter=0
        
        for line in lines[1:2]:
            counter+=1
            get_all_numbers=re.finditer(r"(-)?(\d+\.)?\d+(e(\+|-)\d+)?(?=,)",line)
            numbers=[float(x.group()) for x in get_all_numbers]
            
            params=numbers[1:-1]
            """harms=list(range(3, 10))
            plot_dict=dict(data_data={"time":time, "current":current*1e6, "harmonics":harms}, hanning=True,plot_func=np.abs)
            for k in [10, 5000]:
                params[2]=k
                simcurrent=simulator.dim_i(simulator.Dimensionalsimulate(params, time))*1e6
                plot_dict["{0}_data".format(params[2])]={"time":time,"current":simcurrent, "harmonics":harms}
            sci.plot.plot_harmonics(**plot_dict)
            plt.show()"""
            k_val=params[2]
            k_values=np.logspace(0, np.log10(k_val), 50)
            scores=np.zeros(len(k_values))
            for p in range(0, len(k_values)):
                params[2]=k_values[p]
                simcurrent=simulator.dim_i(simulator.Dimensionalsimulate(params, time))
                simFT=simulator.experiment_top_hat(time, simcurrent)
                scores[p]=sci._utils.RMSE(FT, simFT)
            plt.loglog(k_values, scores, label="{0} Hz".format(hz_values[i]))
plt.xlabel("$k^0$ $s^{-1}$")
plt.ylabel("Score")
plt.show()

            
           