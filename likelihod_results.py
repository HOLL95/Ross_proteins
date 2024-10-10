import numpy as np
import matplotlib.pyplot as plt
import os
import Surface_confined_inference as sci
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
loc="Profile_likelihoods"
freqs=["3_Hz", "9_Hz","15_Hz"]
files=os.listdir(loc)
options=["log","identity"]
normalised=["normalised","raw"]
for p in range(0, 1):
    for z in range(0, 1):
        fig, ax = plt.subplots(1,3)

        for i in range(0, len(freqs)):
            file=[x for x in files if freqs[i] in x][0]
            data=np.loadtxt(os.path.join(loc,file))
            
            r_vals=sorted(np.unique(data[:,0]))
            if options[p]=="log":
                data[:,2]=np.log10(np.abs(data[:,2]))
            min_score=min(data[:,2])
            max_score=max(data[:,2])
            if normalised[z]=="normalised":
                scores=[1-sci._utils.normalise(x, [min_score, max_score]) for x in data[:,2]]
                factor=100
            else:
                scores=data[:,2]
                factor=1
            datadict={key:{} for key in r_vals}
            for j in range(0, len(data[:,0])):
                datadict[data[j,0]][data[j,1]]=scores[j]#data[j,2]
            k_vals=sorted(datadict[r_vals[0]].keys())
            results_array=np.zeros((len(r_vals), len(k_vals)))
            for k in range(0, len(r_vals)):
                for m in range(0, len(k_vals)):
                    results_array[k, m]=datadict[r_vals[k]][k_vals[m]]
            
            X, Y = np.meshgrid(r_vals, k_vals)
            Z=results_array*factor#np.log10(abs(results_array))
            CS=ax[i].contourf(X,Y,Z, 50,cmap=cm.viridis)
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('top', size='5%', pad=0.05)
            
            fig.colorbar(CS, cax=cax, orientation='horizontal')

            cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            
            ax[i].axhline(90, color="black", linestyle="--")
            ax[i].axhline(1000, color="black", linestyle="--")
            if i!=0:
                ax[i].axvline(330, color="black", linestyle="--")
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            ax[i].set_ylabel("$k^0$ ($s^{-1}$)")
            ax[i].set_xlabel("$R_u$ ($\\Omega$)")
            split_title=freqs[i].split("_")
            cax.set_title(" ".join(split_title))
        plt.subplots_adjust(top=0.913,
                            bottom=0.098,
                            left=0.043,
                            right=0.991,
                            hspace=0.2,
                            wspace=0.148)
        fig.set_size_inches(15, 7)
        plt.show()
        fig.savefig("Profile_likelihoods/{0}_{1}.png".format(options[p],normalised[z]))