import numpy as np
import matplotlib.pyplot as plt
import os
import Surface_confined_inference as sci
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
loc="Profile_likelihoods"
freqs=["3_Hz", "9_Hz","15_Hz"]
files=os.listdir(loc)
options=["log","identity"]
normalised=["normalised","raw"]
for p in range(0, 1):
    for z in range(0, 1):
        fig, ax = plt.subplots(3,1)

        for i in range(0, len(freqs)):
            
            file=[x for x in files if freqs[i] in x and ".txt" in x][0]
            
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
            #results_array=results_array.T
            X, Y = np.meshgrid(k_vals, r_vals)
            
            Z=results_array*factor#np.log10(abs(results_array))
            CS=ax[i].contourf(X,Y,Z, 15,cmap=cm.viridis_r)
            current_ytick=list(ax[i].get_ylim())
            """if i==0:
                existing_ytick=current_ytick
            else:
                if current_ytick[0]>existing_ytick[0]:
                    existing_ytick[0]=current_ytick[0]
           """
            for axis in ax:
                axis.set_xlim([52, current_ytick[1]])
            #divider = make_axes_locatable(ax[i])
           
            if i==0:
                cax=fig.add_axes([0.05,0.05,0.85,0.02])
                #cax = divider.append_axes('top', size='5%', pad=0.05)
                
                fig.colorbar(CS, cax=cax, orientation='horizontal')
                cax.xaxis.set_major_formatter(ticker.PercentFormatter())
                cax.set_xlabel(r"Normalised fraction of $\log|\mathcal{L}(\theta | I(t))|$")
                #cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                
            
            ax[i].axvline(19, color="black", linestyle="--")
            ax[i].axvline(173, color="black", linestyle="--")
            if i!=0:
                ax[i].axhline(330, color="black", linestyle="--")
            ax[i].set_xscale("log")
            ax[i].set_yscale("log")
            if i==2:
                ax[i].set_xlabel("$k^0$ ($s^{-1}$)")
            ax[i].set_ylabel("$R_u$ ($\\Omega$)")
            split_title=freqs[i].split("_")
            ax[i].set_title(" ".join(split_title))
        plt.subplots_adjust(top=0.968,
                            bottom=0.148,
                            left=0.248,
                            right=0.985,
                            hspace=0.2,
                            wspace=0.148)
        fig.set_size_inches(4, 12)
        plt.show()
        fig.savefig("Profile_likelihoods/{0}_{1}.png".format(options[p],normalised[z]), dpi=500)