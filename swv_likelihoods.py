import numpy as np
import matplotlib.pyplot as plt
import os
import Surface_confined_inference as sci
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
freqs=[150, 200, 250, 300, 350, 400, 450, 500]
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
options=["log","identity"]
dirlabels=["A","C"]
params=["E0_mean","E0_std","gamma", "Cdl","CdlE1","alpha"]#

loc="Profile_likelihoods/SWV"
files=os.listdir(loc)

normalised=["normalised","raw"]
p=0
z=0
all_titles=sci._utils.get_titles(params+["k0"])

#fig, ax = plt.subplots(1,3)
log_params=["gamma"]
cmap = plt.get_cmap('viridis_r')
format_params=["Cdl","gamma"]
fig, axes=plt.subplots(len(freqs)*2, len(params))

axes[0,0].text(-0.25, 1.28, all_titles[-1],
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[0,0].transAxes, fontweight="bold")
plt.subplots_adjust(top=0.968,
bottom=0.15,
left=0.115,
right=0.966,
hspace=0.2,
wspace=0.2)
fig.set_size_inches(7, 15)
for i in range(0, len(freqs)):
    axis=fig.add_subplot(len(freqs),len(params),(i*len(params))+1)
    axis.set_ylabel(strfreqs[i]+" Hz", labelpad=40)
    #axis.set_visible(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor('none')

for i in range(0, len(freqs)):
    for j in range(0, len(directions)):
        for param_counter_1 in range(0, len(params)):
            
            ax=axes[(i*2)+j, param_counter_1]
            param1=params[param_counter_1]
            if param1=="k0":
                continue
            param2="k0"
            kr_check=[param1, param2]
    
            
            
            for f in files:
                splitfile=f.split("-")
                if param1 in splitfile and strfreqs[i] in f and directions[j] in f:
                    file=f 
                    break
            
      
            data=np.loadtxt(os.path.join(loc,file))
            Y_vals=sorted(np.unique(data[:,1]))
            X_vals=sorted(np.unique(data[:,0]))

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
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
            datadict={key:{} for key in X_vals}
            for g in range(0, len(data[:,0])):
                datadict[data[g,0]][data[g,1]]=scores[g]#data[j,2]
            
            results_array=np.zeros((len(X_vals), len(Y_vals)))
            for k in range(0, len(X_vals)):
                for m in range(0, len(Y_vals)):
                    results_array[k, m]=datadict[X_vals[k]][Y_vals[m]]
         
            results_array=results_array.T
            X, Y = np.meshgrid(X_vals, Y_vals)
            Z=results_array*factor#np.log10(abs(results_array))
            
            CS=ax.contourf(X,Y,Z, 15,cmap=cm.viridis_r)
            if param1 in log_params:
                ax.set_xscale("log")
            
            ax.yaxis.set_major_formatter(lambda x, pos:"$10^{%.1f}$"% np.log10(x) if x>0 else "0" )
            if param_counter_1!=0:
                ax.set_yticks([])
          
            if param_counter_1==len(params)-1:
                twinx=ax.twinx()
                twinx.set_ylabel(dirlabels[j], rotation=0)
                twinx.set_yticks([])
            if (i*2)+j!=(len(freqs)*2)-1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(all_titles[param_counter_1])
            if i==0 and param_counter_1==0 and j==0:
                colorax = fig.add_axes([0.45,0.03,0.35,0.025])
                fig.colorbar(CS, cax=colorax, orientation='horizontal')
                colorax.set_xlim([0, 100])
                colorax.text(-0.55, 0.5, r"Normalised fraction of $\log|\mathcal{L}(\theta | I(t))|$",
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=colorax.transAxes)
               
                colorax.xaxis.set_major_formatter(ticker.PercentFormatter())
                

            """
            
            if param1 in format_params:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
            if param2 in format_params:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        
            if param_counter_1!=len(params)-1:
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_xlabel(all_titles[param_counter_2])
                
                other_val=best_fits[freqs[i]][full_list.index(params[-1])]
                single_slice=np.array([datadict[x][other_val]for x in X_vals])
                
                plot_ax=axes[param_counter_2, param_counter_2].twinx()
                axes[param_counter_2, param_counter_2].set_yticks([])
                plot_ax.scatter(X_vals, single_slice*100, color=cmap(single_slice,10), s=5)
                if param2 in log_params:
                    plot_ax.set_xscale("log")
                plot_ax.yaxis.set_major_formatter(ticker.PercentFormatter())
                plot_ax.xaxis.set_minor_formatter(NullFormatter())
                plot_ax.xaxis.set_major_formatter(NullFormatter())
                #plot_ax.yaxis.set_minor_formatter(NullFormatter())
                #plot_ax.yaxis.set_major_formatter(NullFormatter())

                if param_counter_2==0:
                    other_val=best_fits[freqs[i]][full_list.index(params[0])]
                    single_slice=np.array([datadict[other_val][x] for x in Y_vals])
                    plot_ax=axes[-1, -1].twinx()
                    axes[-1, -1].set_yticks([])
                    plot_ax.scatter(Y_vals, single_slice*100, color=cmap(single_slice, 10), s=5)
                    print(all_titles[-1])
                    axes[-1, -1].set_xlabel(all_titles[-1])
                    if params[-1] in log_params:
                        plot_ax.set_xscale("log")
                    #plot_ax.set_ylim([0, 105])
                    plot_ax.yaxis.set_major_formatter(ticker.PercentFormatter())
                    
                    #plot_ax.yaxis.set_minor_formatter(NullFormatter())
                    #plot_ax.yaxis.set_major_formatter(NullFormatter())
                
            if param_counter_2!=0:
                ax.set_yticks([])
                ax.yaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylabel(all_titles[param_counter_1])
            
                
            #else:
            #    ax.set_xlabel(param2)
            #divider = make_axes_locatable(ax[i])
            #cax = divider.append_axes('top', size='5%', pad=0.05)
            
            #

            #cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            
            #ax[i].axhline(90, color="black", linestyle="--")
            #ax[i].axhline(1000, color="black", linestyle="--")
            #if i!=0:
            #    ax[i].axvline(330, color="black", linestyle="--")
            #ax[i].set_xscale("log")
            #ax[i].set_yscale("log")
            #ax[i].set_ylabel("$k^0$ ($s^{-1}$)")
            #ax[i].set_xlabel("$R_u$ ($\\Omega$)")
            #split_title=freqs[i].split("_")
            #cax.set_title(" ".join(split_title))"""
        
fig.set_size_inches(7, 15)
plt.show()
fig.savefig("Profile_likelihoods/SWV_grid.png".format(freqs[i]), dpi=500)