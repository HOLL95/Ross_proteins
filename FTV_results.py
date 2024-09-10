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
for i in range(0, len(hz_values)):
    for j in range(0, len(amp_values)):
        desired_file="{0}Hz_FTV_Fourier_2_{1}".format(hz_values[i], amp_values[j])
        if desired_file in folders:
            scatter_folders.append(desired_file)
            labels.append("{0} Hz, {1} mV".format(hz_values[i], amp_values[j]))
fig, ax=plt.subplots()
for i in range(0, len(scatter_folders)):
    k_r=[]
   
    
    pooled_loc=loc+"/"+scatter_folders[i]
    pooled_files=os.listdir(pooled_loc)
    pooled_name=[x for x in pooled_files if "Pooled" in x][0]
    table_loc="/".join([pooled_loc, pooled_name, "Full_table.txt"])
    with open(table_loc, "r") as f:
        lines=f.readlines()
        params=lines[0].split(",")
        Ru_idx=[x for x in range(0, len(params)) if "R_u" in params[x]][0]
        k0_idx=[x for x in range(0, len(params)) if "k_0" in params[x]][0]
        
        for line in lines[1:]:
            get_all_numbers=re.finditer(r"(-)?(\d+\.)?\d+(e(\+|-)\d+)?(?=,)",line)
            numbers=[float(x.group()) for x in get_all_numbers]
            k_r.append([numbers[k0_idx], numbers[Ru_idx], numbers[0]])
    k_r=np.array(k_r)
    ax.scatter(k_r[:,0], k_r[:,1], s=np.flip(k_r[:,2])*5, label=labels[i])
    ax.scatter(k_r[0,0], k_r[0,1], s=100, edgecolor="black", facecolor="None")
ax.set_yscale("log")
ax.set_xscale("log")
ax.axvline(120, color=sci._utils.colours[0], linestyle="--", label="3Hz Reversibility")
ax.axvline(285, color=sci._utils.colours[2], linestyle="--", label="9Hz Reversibility")
ax.axvline(380, color=sci._utils.colours[4], linestyle="--", label="15Hz Reversibility")
#ax.axvline(1.0608153188e+03, color="black", linestyle="--", label="3Hz PSV $k_0$")
#ax.axhline(1.6242309585e+03, color="black", linestyle="--", label="3Hz PSV $R_u$")
ax.axhline(1402.3641816809072, color="black", linestyle=":", label="EIS $R_u$")

plt.legend()
ax.set_xlabel("$k_0$ $(s^{-1})$")
ax.set_ylabel("$R_u$ $(\\Omega)$")
plt.show()