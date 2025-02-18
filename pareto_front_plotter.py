import numpy as np
import matplotlib.pyplot as plt
from ax.plot.pareto_frontier import plot_pareto_frontier
import os
import itertools
import Surface_confined_inference as sci
loc="frontier_tests/frontier_results/set2"
   
files = os.listdir(loc)
linestart=len(r"Parameterization:</em><br>")
parameters=['E0_mean', 'E0_std', 'k0', 'gamma', 'alpha']
combinations=[('E0_mean', 'E0_std'), ('E0_mean', 'k0'), ('E0_mean', 'gamma'), ('E0_mean', 'alpha'), ('E0_std', 'k0'), ('E0_std', 'gamma'), ('E0_std', 'alpha'), ('k0', 'gamma'), ('k0', 'alpha'), ('gamma', 'alpha')]
freqs=[65, 75,85, 100, 115, 125, 135]
colours=["blue","orange","green","red","pink","purple", "black"]
freqcolours=dict(zip(freqs, colours))
fig, axis=plt.subplots(2,5)
boundaries={
    "E0":[-0.5, -0.3],
    "E0_mean":[-0.5, -0.3],
    "E0_std":[1e-3, 0.1],
    "k0":[0.1, 5e3],
    "alpha":[0.4, 0.6],
    "gamma":[1e-11, 1e-9],
    "Cdl":[-10,10],
    "CdlE1":[-10, 10],
    "CdlE2":[-10, 10],
    "CdlE3":[-10, 10],
    "alpha":[0.4, 0.6],
    "E0_offset":[0, 0.2],
    }
for file in files:
    filed=file[:file.index(".")]
    splitfile=filed.split("_")
    freq1=int(splitfile[0])
    freq2=int(splitfile[1])
    results_dict=np.load(os.path.join(loc, file), allow_pickle=True).item()["frontier"]
    values=plot_pareto_frontier(results_dict, CI_level=0.90)
    text=values.data["data"][0]["text"]
    for line in text:
        starting_idx=line.find("Parameterization:")
        parameter_list=line[starting_idx+linestart:].split("<br>")
        param_dict={}
        for element in parameter_list:
            splitelement=element.split(":")
            param_dict[splitelement[0]]=float(splitelement[1])
        
        un_normalised_dict={key:sci._utils.un_normalise(param_dict[key], boundaries[key]) for key in param_dict.keys()}
        for j in range(0, len(combinations)):
            ax=axis[j//5, j%5]
            ax.scatter(un_normalised_dict[combinations[j][0]],un_normalised_dict[combinations[j][1]], c=freqcolours[freq1], edgecolors=freqcolours[freq2])
            if file==files[0]:
                ax.set_xlabel(combinations[j][0])
                ax.set_ylabel(combinations[j][1])
    
fig.set_size_inches(12, 12)
plt.tight_layout()
fig.savefig("Initial_pareto_scatter_plots.png", dpi=500)