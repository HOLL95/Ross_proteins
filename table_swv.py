import numpy as np
import matplotlib.pyplot as plt
import Surface_confined_inference as sci
import os
loc="/home/henryll/Documents/Experimental_data/Nat/m4d2_SET3"
files =os.listdir(loc)
results={"anodic":{},"cathodic":{}}
freqs=[65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
strfreqs=[str(x) for x in freqs]
labels=["linear","quadratic","cubic"]

directions=["anodic","cathodic"]
params=["E0_mean","E0_std","k0","gamma"]
colours=["orange", "red", "green", "orange"]
loc="/home/henryll/Documents/Inference_results/swv/set4_13"
loc2="/home/henryll/Documents/Inference_results/swv/set4_14"
loc="/home/henryll/Documents/Inference_results/swv/set6_1"
ylabels=[r"$E^0 \mu$ (V)", r"$E^0 \sigma$ (V)", r"$k_0$ ($s^{-1}$)", r"$\Gamma$ (mol cm$^{-2}$)"]
#values=dict(zip(params, [np.zeros((len(labels),len(freqs))) for x in params]))
fig, ax=plt.subplots(2,2)
fac=10
vals=[-1*fac, fac]
markers=["o","^"]

table_dict={}
for i in range(0, len(freqs)):
    f_key="{0} Hz".format(freqs[i])
    table_dict[f_key]={"anodic":{}, "cathodic":{}}
    for j in range(0, len(directions)):
        table_dict[f_key][directions[j]]={x:{} for x in labels}
        for m in range(0, len(labels)):
            
               
            param_values, titles=sci._utils.read_param_table(os.path.join(loc, "SWV_{0}".format(freqs[i]), directions[j], labels[m], "PooledResults_2024-11-18","Full_table.txt"), get_titles=True)
            rounded=[0 for x in range(0, len(param_values[0]))]
           
            for o in range(0, len(rounded)):
                if "E^0" in titles[o] or "Dimensionless" in titles[o]:

                    rounded[o]=sci._utils.format_values(param_values[0][o],3)
                else:
                    rounded[o]=sci._utils.format_values(param_values[0][o],2)
            table_dict[f_key][directions[j]][labels[m]]=dict(zip(titles, rounded))


split_list=list(table_dict["65 Hz"]["anodic"]["cubic"].keys())
print("Frequency (Hz) & "+" & ".join(split_list)+r"\\ \hline")
for i in range(0, len(freqs)):
    f_key="{0} Hz".format(freqs[i])
    
    for j in range(0, len(directions)):
        empty_list=[[] for x in range(0, len(split_list))]
        for z in range(0, len(labels)):
            curr_keys=list(table_dict[f_key][directions[j]][labels[z]].keys())
            
            for m in range(0, len(split_list)):
                
                if split_list[m] in curr_keys:
                    empty_list[m].append(str(table_dict[f_key][directions[j]][labels[z]][split_list[m]]))
                else:
                    empty_list[m].append("-")
        for m in range(0, len(empty_list)):
            empty_list[m]="/".join(empty_list[m])
        print(" & ".join(["{1} ({0})".format(directions[j][0].upper(), f_key)]+ empty_list)+r"\\ \hline")
            