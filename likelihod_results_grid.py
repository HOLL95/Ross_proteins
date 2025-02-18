import numpy as np
import matplotlib.pyplot as plt
import os
import Surface_confined_inference as sci
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
freqs=["3_Hz", "9_Hz","15_Hz"]

options=["log","identity"]
params=["E0_mean","E0_std","k0","gamma", "Ru","Cdl","omega","alpha"]#"CdlE1","CdlE2","CdlE3",
full_list=["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","alpha"]
best_fits=dict(zip(freqs,[
    [-0.4244818992, 0.0417143285,     4.9920932937e+03, 1.8134851259e-10,         2.5644285241e+03, 6.6014432067e-05, -4.6443012010e-03, -1.2340727286e-03, -1.4626502188e-05, 3.0346809949,  0.524292126,],
    [-0.4205770335, 0.0650104188,     1.8457772599e+03, 1.1283378372e-10,         764.9843288034,   9.9054277034e-05, -3.5929255146e-03, -7.1997893984e-04, -5.3937670998e-06, 9.1041692039,  0.4770065713,     ],
    [-0.4225575542, 0.0765111705,     111.5784183264,   7.7951293716e-11,         184.9121615528,   9.9349322658e-05, -1.6383464471e-03, -3.3463983363e-04, -7.5051594548e-06, 15.1735928635, 0.4006210586, ]

]))
files=os.listdir("Profile_likelihoods/collated")
normalised=["normalised","raw"]
p=0
z=0
all_titles=sci._utils.get_titles(params)

#fig, ax = plt.subplots(1,3)

cmap = plt.get_cmap('viridis_r')
format_params=["Cdl","gamma"]
for i in range(0, len(freqs)):
    fig, axes=plt.subplots(len(params), len(params))
    if i==1:
        log_params=["k0",]
    else:
        log_params=["k0","Ru"]

    for param_counter_1 in range(0, len(params)):
        for param_counter_2 in range(0, len(params)):
            ax=axes[param_counter_1, param_counter_2]
            param1=params[param_counter_1]
            param2=params[param_counter_2]
            kr_check=[param1, param2]
            
            if param_counter_1==param_counter_2:
                
                continue
            elif param_counter_1>param_counter_2:
                if ("k0" in kr_check) and ("Ru" in kr_check):
                    loc="Profile_likelihoods"
                    file="{0}_scan_results.txt".format(freqs[i])
                    data=np.loadtxt(os.path.join(loc,file))
                    Y_vals=sorted(np.unique(data[:,1]))
                    X_vals=sorted(np.unique(data[:,0]))
                else:
                    loc="Profile_likelihoods/collated"
                    file= [x for x in files if (freqs[i] in x and param1 in x and param2 in x)]
                    if "Cdl" in kr_check:
                        for f in file:
                            removed_end=f[:f.index(".")]
                            split_file=removed_end.split("_")
                           
                            if "Cdl" in split_file:
                                file=f
                    else:
                        file=file[0]
                    data=np.loadtxt(os.path.join(loc,file))
                    if param1=="gamma" or param1=="Cdl":
                        y_factor=(0.07/0.036)
                    else:
                        y_factor=1
                    if param2=="gamma" or param2=="Cdl":
                        x_factor=(0.07/0.036)
                    else:
                        x_factor=1
                    Y_vals=sorted(np.unique(data[:,1]))
                    X_vals=sorted(np.unique(data[:,0]))
          
            else:
                ax.set_axis_off()
                continue
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
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
            for j in range(0, len(data[:,0])):
                datadict[data[j,0]][data[j,1]]=scores[j]#data[j,2]
            
            results_array=np.zeros((len(X_vals), len(Y_vals)))
            for k in range(0, len(X_vals)):
                for m in range(0, len(Y_vals)):
                    results_array[k, m]=datadict[X_vals[k]][Y_vals[m]]
            if ("k0" in kr_check) and ("Ru" in kr_check):
                pass 
            else:
                results_array=results_array.T
            X, Y = np.meshgrid(X_vals, Y_vals)
            Z=results_array*factor#np.log10(abs(results_array))
            if param1 in log_params:
                ax.set_yscale("log")
            if param2 in log_params:
                ax.set_xscale("log")
            CS=ax.contourf(X*x_factor,Y*y_factor,Z, 10,cmap=cm.viridis_r)
            
            if param1 in format_params:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
            if param2 in format_params:
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
           
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
            if param_counter_2==0 and param_counter_1==1:
                colorax = fig.add_axes([0.55,0.75,0.35,0.05])
                fig.colorbar(CS, cax=colorax, orientation='horizontal')
                colorax.set_xlabel(r"Normalised fraction of $\log|\mathcal{L}(\theta | I(t))|$")
                colorax.xaxis.set_major_formatter(ticker.PercentFormatter())
                colorax.set_title(" ".join(freqs[i].split("_")))
                
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
            #cax.set_title(" ".join(split_title))
    plt.subplots_adjust(top=0.968,
                        bottom=0.098,
                        left=0.065,
                        right=0.941,
                        hspace=0.2,
                        wspace=0.148)
    fig.set_size_inches(15, 15)
    #plt.show()
    fig.savefig("Profile_likelihoods/{0}_grid.png".format(freqs[i]), dpi=500)