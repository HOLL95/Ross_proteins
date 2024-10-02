import numpy as np
import matplotlib.pyplot as plt
import Surface_confined_inference as sci
import os
freqs=[100, 150, 200, 250, 300, 350, 400, 450, 500]
strfreqs=[str(x) for x in freqs]
labels=["constant", "linear","quadratic","cubic"]
directions=["anodic","cathodic"]
params=["E0_mean","E0_std","k0","gamma"]
colours=["blue", "orange","green","red"]
loc="/home/henryll/Documents/Inference_results/swv/set4_4"
ylabels=[r"$E^0 \mu$ (V)", r"$E^0 \sigma$ (V)", r"$k_0$ ($s^{-1}$)", r"$\Gamma$ (mol cm$^{-2}$)"]
#values=dict(zip(params, [np.zeros((len(labels),len(freqs))) for x in params]))
fig, ax=plt.subplots(2,2)
fac=10
vals=[-1*fac, fac]
markers=["o","^"]
for i in range(0, len(freqs)):
    if i==1:
        

        for z in range(0, len(params)):
            axis=ax[z//2, z%2]
            axis.set_ylabel(ylabels[z])
            axis.set_xlabel("Frequency (Hz)")
            axis.set_xticks(freqs)
                
    for j in range(0, len(directions)):
        for m in range(0, len(labels)):
            param_values=sci._utils.read_param_table(os.path.join(loc, "SWV_{0}".format(freqs[i]), directions[j], labels[m], "PooledResults_2024-09-26","Full_table.txt"))[0]
            for z in range(0, len(params)):
                axis=ax[z//2, z%2]
                axis.scatter(freqs[i]+vals[j], param_values[z], color=colours[m], marker=markers[j])
plots=[{"options":{"color":colours}, "labels":labels}, {"options":{"marker":markers}, "labels":directions}]
for i in range(0,2):
    twinx=ax[0,i].twinx()
    twinx.set_yticks([])
    key=list(plots[i]["options"].keys())[0]
    for j in range(0, len(plots[i]["options"][key])):
        if i==0:
            
            twinx.scatter(0, None, label=plots[i]["labels"][j], **{key:plots[i]["options"][key][j]})
        else:
            twinx.scatter(0, None, label=plots[i]["labels"][j],color="black", **{key:plots[i]["options"][key][j]})
    twinx.legend(loc="center", bbox_to_anchor=[0.5, 1.05], ncols=len(plots[i]["options"][key]), frameon=False)


ax[1,0].axhline(120, color="black", linestyle="--", label="15 Hz FTACV value")
ax[1,0].legend()
plt.show()