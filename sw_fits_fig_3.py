import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import os
from matplotlib import cm
import matplotlib.ticker as ticker
loc="/home/userfs/h/hll537/Documents/Experimental_data/SWV"
loc="/users/hll537/Experimental_data/SWV"
loc="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/SquareWave"
freqs=[200,  450]
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
files=os.listdir(loc)
paramloc="/home/henryll/Documents/Inference_results/swv/set4_9"
likelihood_params=["E0_mean", "gamma"]
from matplotlib.patches import Rectangle
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.ticker import MaxNLocator
fig, axes=plt.subplots(2, 2)
axes[0, 1].set_axis_off()
axes[1, 1].set_axis_off()
likelihood_ax=[["" for x in range(0, 2)] for y in range(0, 4)]
for i in range(0, 4):
    for j in range(0, 2):
        likelihood_ax[i][j]=fig.add_subplot(4, 4, (i*4)+3+j)

plt.subplots_adjust(top=0.93,
bottom=0.215,
left=0.103,
right=0.975,
hspace=0.355,
wspace=0.265)
profileloc="Profile_likelihoods/SWV"
profilefiles=os.listdir(profileloc)

all_titles=sci._utils.get_titles(likelihood_params+["k0"])
likelihood_ax[0][0].text(1.1, 1.15, all_titles[-1],
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[0,0].transAxes)

for i in range(0,len(freqs)):
    
    for j in range(0, len(directions)):
        
        
        file=[x for x in files if (strfreqs[i] in x and directions[j] in x)][0]
        print(os.path.join(loc, file))
        try:
            data=np.loadtxt(os.path.join(loc, file), delimiter=",")
        except:
            data=np.loadtxt(os.path.join(loc, file))
        pot=data[:-1,0]
        data_current=data[:-1,  2]*1e6
        if j==0:
            label="Data"
        else:
            label=None
        ax=axes[i,0]
        ax.set_title(strfreqs[i]+" Hz")
        ax.plot(pot,data_current, label=label, color=sci._utils.colours[0])
        sw_class=sci.SingleSlurmSetup("SquareWave",
                                    {"omega":freqs[i],
                                    "scan_increment":2e-3,#abs(pot[1]-pot[0]),
                                    "delta_E":0.8,
                                    "SW_amplitude":2e-3,
                                    "sampling_factor":200,
                                    "E_start":directions_dict[directions[j]]["E_start"],
                                    "Temp":278,
                                    "v":directions_dict[directions[j]]["v"],
                                    "area":0.07,
                                    "N_elec":1,
                                    "Surface_coverage":1e-10},
                                    square_wave_return="net",
                                    problem="forwards"

                                    )
      
        times=sw_class.calculate_times()
        potential=sw_class.get_voltage(times)
        
        ax.set_ylabel("Current ($\\mu A$)", labelpad=7)
            
        if i!=0:
            ax.set_xlabel("Potential (V)", labelpad=7)

        extra_keys=["CdlE1","CdlE2","CdlE3"]
        labels=["Constant", "Linear","Quadratic","Cubic"]
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
        #newfilelist=file.split(".")
        #newfilelist[-1]="txt"
        #savefile=".".join(newfilelist)
        #np.savetxt(os.path.join(saveloc, savefile),np.column_stack((sw_class._internal_memory["SW_params"]["b_idx"], data_current, pot)),)
        for m in range(2, 3):

            
            #sw_class.fixed_parameters={"alpha":0.5}
           
            try:
                best_fit=sci._utils.read_param_table(os.path.join(paramloc, "SWV_{0}".format(freqs[i]), directions[j], labels[m].lower(), "PooledResults_2024-10-30","Full_table.txt"))[0]
            except:
                continue
            
            sw_class.dispersion_bins=[50]
            sw_class.optim_list=["E0_mean","E0_std","k0","gamma","Cdl"]+extra_keys[:m]+["alpha"]
            sim=1e6*sw_class.dim_i(sw_class.Dimensionalsimulate(best_fit[:-1], times))
            if j==0:                  
                ax.plot(pot, sim, color=sci._utils.colours[1], label="Simulation", )
            else:
                 ax.plot(pot, sim, color=sci._utils.colours[1])
       
        for m in range(0, len(likelihood_params)):
            
            ax=likelihood_ax[(i*2)+j][m]
            

            param1=likelihood_params[m]
            
            for f in profilefiles:
                splitfile=f.split("-")
                if param1 in splitfile and strfreqs[i] in f and directions[j] in f:
                    file=f 
                    break
            score_data=np.loadtxt(os.path.join(profileloc,file))
            
            Y_vals=sorted(np.unique(score_data[:,1]))
            X_vals=sorted(np.unique(score_data[:,0]))
            score_data[:,2]=np.log10(np.abs(score_data[:,2]))
            min_score=min(score_data[:,2])
            max_score=max(score_data[:,2])
            scores=[1-sci._utils.normalise(x, [min_score, max_score]) for x in score_data[:,2]]
            print(min(scores), max(scores))
            factor=100
            datadict={key:{} for key in X_vals}
            for g in range(0, len(score_data[:,0])):
                datadict[score_data[g,0]][score_data[g,1]]=scores[g]#data[j,2]
            
            results_array=np.zeros((len(X_vals), len(Y_vals)))
            for k in range(0, len(X_vals)):
                for r in range(0, len(Y_vals)):
                    results_array[k, r]=datadict[X_vals[k]][Y_vals[r]]
            
            results_array=results_array.T
            X, Y = np.meshgrid(X_vals, Y_vals)
            Z=results_array*factor
            CS=ax.contourf(X,Y,Z, 15,cmap=cm.viridis_r)
            if param1 =="gamma":
                    ax.set_xscale("log")
          
            
            if (i*2)+j==3:    
                xlims=[-100, 1e25]
                likelihood_ax[3][m].set_xlabel(all_titles[m])
                
                for q in range(0, 3):
                    xlim=likelihood_ax[q][m].get_xlim()
                    xlims[0]=max(xlims[0], xlim[0])
                    xlims[1]=min(xlims[1], xlim[1])
                for q in range(0, 4):
                    if m==1:
                        print(xlims)
                    likelihood_ax[q][m].set_xlim(xlims)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                ax.set_xticks([])
            if m!=0:
                ax.set_yticks([])
            if i==0 and j==0 and m==0:
                colorax = fig.add_axes([0.1,0.07,0.35,0.035])
                fig.colorbar(CS, cax=colorax, orientation='horizontal')
                colorax.xaxis.set_major_formatter(ticker.PercentFormatter())
                colorax.set_xlim([0, 100])
                #colorax.set_ylabel(r"Normalised fraction of $\log|\mathcal{L}(\theta | I(t))|$",)

                
                    

            
                
            ax.yaxis.set_major_formatter(lambda x, pos:"$10^{%.1f}$"% np.log10(x) if x>0 else "0" )

axes[0,0].legend()
fig.set_size_inches(8.5, 4.5)
plt.show()
fig.savefig("fig3_draft.png",dpi=500)

            
