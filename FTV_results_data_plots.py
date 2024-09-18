import numpy as np
import matplotlib.pyplot as plt
import os
import re
import Surface_confined_inference as sci
loc="/home/henryll/Documents/Inference/M4D2/set9"
folders=os.listdir(loc)
hz_values=["3", "9", "15", "21"]
amp_values=["80", "250"]
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
plots=sci.plot.multiplot(orientation="landscape", num_rows=3, num_cols=2, num_harmonics=n_harms, harmonic_position=[0,1,2], row_spacing=2, col_spacing=1, plot_width=4)
axes=plots.axes_dict
factor=1e9
for i in range(0, len(scatter_folders)):#
    k_r=[]
    rowdx=(i//2)+1
    row=axes["row{0}".format(rowdx)]
    print(len(row), "row{0}".format(rowdx))
    rowdx2=i%2
    row_axes=row[rowdx2*n_harms:(rowdx2+1)*n_harms]

    pooled_loc=loc+"/"+scatter_folders[i]
    pooled_files=os.listdir(pooled_loc)
    pooled_name=[x for x in pooled_files if "Pooled" in x][0]
    table_loc="/".join([pooled_loc, pooled_name, "Full_table.txt"])
    full_data_loc=data_loc+"/"+index_dict[scatter_folders[i]]["amp"]
    data_file="FTACV_m4D2_PGE_59.60_mV_s-1_{0}_Hz_{1}_mV.txt".format(index_dict[scatter_folders[i]]["freq"],index_dict[scatter_folders[i]]["amp"])
    data=np.loadtxt(full_data_loc+"/"+data_file)
    time=data[::13,0]
    current=data[::13,1]
    potential=data[::13,2]
    simulator=sci.LoadSingleExperiment("FTV_sims/FTV_simulator_{0}_Hz_{1}_mv.json".format(index_dict[scatter_folders[i]]["freq"],index_dict[scatter_folders[i]]["amp"]))

    plot_dict={"Data_data":{"time":time, "current":current*factor, "potential":potential, "harmonics":harmonics}, 
                "hanning":True, 
                "axes_list":row_axes, 
                "plot_func":np.abs, 
                "remove_xaxis":True,
                "title":"{0} Hz, {1} mV".format(index_dict[scatter_folders[i]]["freq"],index_dict[scatter_folders[i]]["amp"]),
                "nyticks":2}
    
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
            k0_val=numbers[k0_idx]
            labels=["Sim", "Reversible"]
            colours=["red","black"]
            linestyles=["-", "--"]
            kvals=[k0_val]
            for m in range(0, len(kvals)):
                numbers[k0_idx]=kvals[m]
                params=numbers[1:-1]

                simcurrent=simulator.dim_i(simulator.Dimensionalsimulate(params, time))*factor
                plot_dict["{0}_data".format(labels[m])]={"time":time, "current":simcurrent, "harmonics":harmonics, "colour":colours[m], "linestyle":linestyles[m]}

    if i==0:
        plot_dict["legend"]={"ncols":2, "bbox_to_anchor":[-0.05, 1.9], "loc":"center", "frameon":False}
    else:
        plot_dict["legend"]=None
    sci.plot.plot_harmonics(**plot_dict)
axes["row3"][n_harms-1].set_xlabel("Time (s)")
axes["row3"][-1].set_xlabel("Time (s)")
axes["row2"][int(n_harms)//2].set_ylabel("Current (nA)")

fig=plt.gcf()
fig.set_size_inches(8.3, 11.7)
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.125, right=0.95, hspace=0.2, wspace=0.2)

fig.savefig("FTV_full_plot.png", dpi=500)
