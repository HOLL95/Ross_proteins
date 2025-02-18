import numpy as np
import matplotlib.pyplot as plt
from ax.plot.pareto_frontier import plot_pareto_frontier
import os
import itertools
import Surface_confined_inference as sci
import matplotlib.ticker as ticker
loc="frontier_tests/frontier_results/set2"
import copy
files = os.listdir(loc)
linestart=len(r"Parameterization:</em><br>")
parameters=['E0_mean', 'E0_std', 'k0', 'gamma', 'alpha']
combinations=[('E0_mean', 'E0_std'), ('E0_mean', 'k0'), ('E0_mean', 'gamma'), ('E0_mean', 'alpha'), ('E0_std', 'k0'), ('E0_std', 'gamma'), ('E0_std', 'alpha'), ('k0', 'gamma'), ('k0', 'alpha'), ('gamma', 'alpha')]
freqs=[65, 75,85, 100, 115, 125, 135]
colours=["blue","orange","green","red","magenta","purple", "black"]
combinations=list(itertools.combinations(freqs,2))
strfreqs=[str(x) for x in freqs]
freqcolours=dict(zip(strfreqs, colours))

ax_client=np.load("/home/henryll/Documents/Ross_protein/frontier_tests/frontier_test_2.npy", allow_pickle=True).item()["saved_frontier"]
thresholds={'65': 0.12264263913778613, '75': 0.12983951585847955, '85': 0.12785030588143828, '100': 0.14377940999664807, '115': np.float64(0.1322970338879717), '125': 0.1353898124271389, '135': 0.11645078613573955}
min_dict={str(x):[thresholds[str(x)],thresholds[str(x)]] for x in freqs}#
fig, axis=plt.subplots(2,4)
plot_dict={}
for i in range(0, len(combinations)):
    
    #ax=axis[i//3, i%3]
    freq1=str(combinations[i][0])
    freq2=str(combinations[i][1])
    key="{0}_{1}".format(freq1, freq2)
    plot_dict[key]={}
    file=[x for x in files if (freq1 in x and freq2 in x)][0]

    """filed=file[:file.index(".")]
    splitfile=filed.split("_")
    freq1=int(splitfile[0])
    freq2=int(splitfile[1])"""
    results_dict=np.load(os.path.join(loc, file), allow_pickle=True).item()["frontier"]
    values=plot_pareto_frontier(results_dict, CI_level=0.90)
    text=values.data["data"][0]["text"]

    x_scores=values.data["data"][0]["x"]
    y_scores=values.data["data"][0]["y"]
    min_dict[freq1][0]=min(min(x_scores), min_dict[freq1][0])
    min_dict[freq2][0]=min(min(y_scores), min_dict[freq2][0])
    x_err=values.data["data"][0]["error_x"]["array"]
    y_err=values.data["data"][0]["error_y"]["array"]
    plot_dict[key][freq1]=x_scores
    plot_dict[key][freq2]=y_scores
    #ax.errorbar(x_scores, y_scores, xerr=x_err, yerr=y_err)
    #ax.plot(x_scores, y_scores)
    #ax.set_xlabel(freq1+" Hz RMSE")
    #ax.set_ylabel(freq2+" Hz RMSE")
for i in range(0, len(freqs)):
    ax=axis[i//4, i%4]
    for j in range(0, len(freqs)):
        
        if freqs[i]==freqs[j]:
            continue
        else:
            for key in plot_dict.keys():
                if strfreqs[i] in key and strfreqs[j] in key:


        
            
                    raw_x=copy.deepcopy(plot_dict[key][strfreqs[i]])
                    raw_y=copy.deepcopy(plot_dict[key][strfreqs[j]])
                    #print(min_dict[freq1], freq1)
                    #print(min_dict[freq2], freq2)
                    
                    normed_x=[100*(1-sci._utils.normalise(x, min_dict[strfreqs[i]])) for x in raw_x]
                    normed_y=[100*(1-sci._utils.normalise(x, min_dict[strfreqs[j]])) for x in raw_y]
                    
                    
                    #print(min(raw_y), min(normed_y),min_dict[strfreqs[j]], np.array(raw_y)[np.where(raw_y<min_dict[strfreqs[j]])])
                    ax.plot(normed_x, normed_y, label=strfreqs[j], color=freqcolours[strfreqs[j]])
                    ax.set_xlabel("Normalised "+strfreqs[i]+" Hz Score")
                    ax.set_ylabel("Normalised score")
                    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
                    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
                    #ax.set_yscale("log")
                    #ax.set_xscale("log")
axis[-1, -1].set_axis_off()
for i in range(0, len(freqs)):
    axis[-1, -1].plot(0, 0, label=strfreqs[i]+" Hz", color=freqcolours[strfreqs[i]])
axis[-1,-1].legend()
fig.set_size_inches(12,6)
plt.subplots_adjust(top=0.975,
bottom=0.097,
left=0.063,
right=0.982,
hspace=0.266,
wspace=0.342)

plt.show()
fig.savefig("Pareto_errors.png", dpi=500)