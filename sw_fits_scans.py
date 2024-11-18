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
dirlabels=["A","C"]
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}

paramloc="/home/henryll/Documents/Inference_results/swv/set4_10"
likelihood_params=["E0_mean", "gamma"]
from matplotlib.patches import Rectangle
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.ticker import MaxNLocator
fig, axes=plt.subplots(2, 2)

axes[0, 1].set_axis_off()
axes[1, 1].set_axis_off()
likelihood_ax=[["" for x in range(0, 2)] for y in range(0, 4)]
bbox=dict(x0=0.53811810154525385, y0=0.5860509554140128, x1=0.91749999999999999, y1=0.935) 
bboxlower=dict(x0=0.53811810154525385, y0=0.16, x1=0.91749999999999999, y1=0.4639490445859873)

topxend=bbox["x1"]
topyend=bbox["y1"]
topxstart=bbox["x0"]
topystart=bbox["y0"]
title1=fig.add_axes([topxstart, topystart, (topxend-topxstart), (topyend-topystart)])
title1.set_title("200 Hz")
title1.set_axis_off()

byend=bboxlower["y1"]

bystart=bboxlower["y0"]
spacing=0.025
title2=fig.add_axes([topxstart, bystart, (topxend-topxstart), (byend-bystart)])
title2.set_title("450 Hz")
title2.set_axis_off()
height=((byend-bystart)-spacing)/2
topheight=((topyend-topystart)-spacing)/2
width=((topxend-topxstart)-spacing)/2
files=os.listdir(loc)
blankfiles=os.listdir(os.path.join(loc, "Blanks"))
for i in range(0, 2):
    for j in range(0, 2):
        for m in range(0, 2):
            if i==0:
                label="top"
                likelihood_ax[(i*2)+j][m]=fig.add_axes([topxstart+(spacing+width)*m, topystart+(spacing+topheight)*j,width, topheight])
            else:
                label="bottom"
                likelihood_ax[(i*2)+j][m]=fig.add_axes([topxstart+(spacing+width)*m, bystart+(spacing+height)*j,width, height])

plt.subplots_adjust(top=0.93,
bottom=0.12,
left=0.083,
right=0.975,
hspace=0.355,
wspace=0.265)
profileloc="Profile_likelihoods/SWV2"
profilefiles=os.listdir(profileloc)

all_titles=sci._utils.get_titles(likelihood_params+["k0"])
likelihood_ax[0][0].text(1.1, 1.15, all_titles[-1],
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[0,0].transAxes)
filelist=[files, blankfiles]
locs=[loc, os.path.join(loc, "Blanks")]
colors=["#29af7f","darkgrey" ]
labels=["Experimental", "PGE"]
gs=[1,0]
for i in range(0,len(freqs)):
    for j in range(0, len(directions)):
        g=1
        file=[x for x in filelist[g] if (strfreqs[i] in x and directions[j] in x)][0]
                        
        try:
            data=np.loadtxt(os.path.join(locs[g], file), delimiter=",")
        except:
            data=np.loadtxt(os.path.join(locs[g], file))
        pot=data[2:-2,0]
        data_current=data[2:-2,  2]*1e6
        if j==0:
            label=labels[g]
        else:
            label=None
        ax=axes[i,0]
        ax.plot(pot,data_current, label=label, color=colors[g])

for i in range(0,len(freqs)):
    
    for j in range(0, len(directions)):

        g=0
        file=[x for x in filelist[g] if (strfreqs[i] in x and directions[j] in x)][0]
        
        try:
            data=np.loadtxt(os.path.join(locs[g], file), delimiter=",")
        except:
            data=np.loadtxt(os.path.join(locs[g], file))
        pot=data[2:-2,0]
        data_current=data[2:-2,  2]*1e6
        if j==0:
            label=labels[g]
        else:
            label=None
        ax=axes[i,0]
        ax.plot(pot,data_current, label=label, color=colors[g])
        ax.set_title(strfreqs[i]+" Hz")
        
        sw_class=sci.SingleSlurmSetup("SquareWave",
                                    {"omega":freqs[i],
                                    "scan_increment":2e-3,#abs(pot[1]-pot[0]),
                                    "delta_E":0.8,
                                    "SW_amplitude":2e-3,
                                    "sampling_factor":200,
                                    "E_start":directions_dict[directions[j]]["E_start"],
                                    "Temp":278,
                                    "v":directions_dict[directions[j]]["v"],
                                    "area":0.036,
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
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        #newfilelist=file.split(".")
        #newfilelist[-1]="txt"
        #savefile=".".join(newfilelist)
        #np.savetxt(os.path.join(saveloc, savefile),np.column_stack((sw_class._internal_memory["SW_params"]["b_idx"], data_current, pot)),)
        for m in range(2, 3):

            
            #sw_class.fixed_parameters={"alpha":0.5}
           
            try:
                best_fit=sci._utils.read_param_table(os.path.join(paramloc, "SWV_{0}".format(freqs[i]), directions[j], labels[m].lower(), "PooledResults_2024-11-08","Full_table.txt"))[0]
                print(best_fit)
                if freqs[i]==200:
                    if directions[j]=="anodic":
                        best_fit[2]=206.0847753491919 
                        best_fit[3]=1.624323718532389e-10
                    else:
                        best_fit[2]=203.35762408918902
                        best_fit[3]=1.6127442195381528e-10 
                elif freqs[i]==450:
                    if directions[j]=="anodic":
                        best_fit[2]=197.32029888918916 
                        best_fit[3]=3.6645505257491716e-10
                    else:
                        best_fit[2]=190.16931311621622
                        best_fit[3]=3.75405016374619e-10
            except:
                continue
            #best_fit[3]*=(0.07/0.036)
            sw_class.dispersion_bins=[50]
            sw_class.optim_list=["E0_mean","E0_std","k0","gamma","Cdl"]+extra_keys[:m]+["alpha"]
            sim=1e6*sw_class.dim_i(sw_class.Dimensionalsimulate(best_fit[:-1], times))
            if j==0:                  
                ax.plot(pot, sim[1:-2], color="#46327e", label="Simulation", )
            else:
                 ax.plot(pot, sim[1:-2], color="#46327e")
       
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
            
            if likelihood_params[m]=="gamma":
                desired_k=100
                Y_vals=np.array(Y_vals)
                X_vals=np.array(X_vals)
                absmin=np.abs(Y_vals-desired_k)
                closest_val=Y_vals[np.where(absmin==min(absmin))][0]
                score_vals=[datadict[x][closest_val] for x in X_vals]
                best_gamma=X_vals[np.where(score_vals==max(score_vals))]
                print(closest_val, best_gamma[0], freqs[i], "hello", max(score_vals))


            for k in range(0, len(X_vals)):
                for r in range(0, len(Y_vals)):
                    results_array[k, r]=datadict[X_vals[k]][Y_vals[r]]
            

            results_array=results_array.T
            X, Y = np.meshgrid(X_vals, Y_vals)
            Z=results_array*factor
            CS=ax.contourf(X,Y,Z, 15,cmap=cm.viridis_r)
            if param1 =="gamma":
                    ax.set_xscale("log")
                    ax.xaxis.set_major_locator(ticker.LogLocator(subs="all"))
          
            
            if (i*2)+j==2:    
                xlims=[-100, 1e25]
                likelihood_ax[-2][m].set_xlabel(all_titles[m])
                
                for q in range(0, 3):
                    xlim=likelihood_ax[q][m].get_xlim()
                    xlims[0]=max(xlims[0], xlim[0])
                    xlims[1]=min(xlims[1], xlim[1])
                for q in range(0, 4):
                    if m==1:
                        print(xlims)
                    likelihood_ax[q][m].set_xlim(xlims)
                
            else:
                
                ax.set_xticklabels([])
            if m!=0:
                
                ax.set_yticklabels([])
                twinx=ax.twinx()
                twinx.set_ylabel(dirlabels[j], rotation=0, labelpad=4   )
                twinx.set_yticks([])
            #else:
            #    ax.yaxis.set_major_formatter(lambda x, pos:"$10^{%.1f}$"% np.log10(x) if x>0 else "0" )
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if i==0 and j==0 and m==0:
                colorax = fig.add_axes([bbox["x1"]+0.019,bboxlower["y0"]+(height)/2,0.015,(bbox["y1"]-bboxlower["y0"])-height])
                fig.colorbar(CS, cax=colorax, orientation='vertical')
                colorax.yaxis.set_major_formatter(ticker.PercentFormatter())
                colorax.set_ylim([0, 100])
                #colorax.set_ylabel(r"Normalised fraction of $\log|\mathcal{L}(\theta | I(t))|$",)

                
                    

            
                
          

axes[0,0].legend()
fig.set_size_inches(8.5, 4.75)
init_x=axes[0, 1].get_position()
final_x=axes[1, -1].get_position()
print(init_x, final_x)
plt.show()
fig.savefig("sw_fig_lowk.png",dpi=500)

            
