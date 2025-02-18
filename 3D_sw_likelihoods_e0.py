import numpy as np
import matplotlib.pyplot as plt
import os
import Surface_confined_inference as sci
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
from matplotlib import animation
import matplotlib.ticker as ticker
freqs=[65, 75, 85, 100, 115, 125, 135,  150,175, 200, 300,500]
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
options=["log","identity"]
dirlabels=["A","C"]
params=["E0_mean","E0_std","gamma", "Cdl","alpha","CdlE1","CdlE2","CdlE3",]#

loc="Profile_likelihoods/SWV_net_cubic_set2"
files=os.listdir(loc)

normalised=["normalised","raw"]
p=0
z=0
all_titles=sci._utils.get_titles(params+["k0"])
fig = plt.figure()

#fig, ax = plt.subplots(1,3)
log_params=["gamma"]
cmap = plt.get_cmap('viridis_r')
format_params=["Cdl","gamma"]

"""axes[0,0].text(-0.25, 1.45, all_titles[-1],
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[0,0].transAxes, fontweight="bold")"""
plt.subplots_adjust(top=0.968,
bottom=0.15,
left=0.115,
right=0.966,
hspace=0.2,
wspace=0.2)
plt.rc('ytick', labelsize=10)
"""for i in range(0, len(freqs)):

    axis=fig.add_subplot(len(freqs),len(params),(i*len(params))+1)
    axis.set_ylabel(strfreqs[i]+" Hz", labelpad=40)
    #axis.set_visible(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor('none')
"""
axes_list=[]
surfaces=[]
def update(frame, axes, surfaces, elevation=30):
    """Update function for animation - rotates all surfaces."""
    for ax in axes:
        ax.view_init(elev=elevation, azim=frame)
    return axes
for i in range(0, len(freqs)):
    for j in range(1, len(directions)):
        for param_counter_1 in [0]:
            ax = fig.add_subplot(2, 6, i+1, projection='3d')
            axes_list.append(ax)
        
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
            X_vals=np.array(sorted(np.unique(data[:,0])))
            
            plt.setp(ax.get_xticklabels(), rotation=55, ha='right')
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
            X_vals=X_vals[np.where(X_vals>9e-11)]
            results_array=np.zeros((len(X_vals), len(Y_vals)))
            for k in range(0, len(X_vals)):
                for m in range(0, len(Y_vals)):
                    results_array[k, m]=datadict[X_vals[k]][Y_vals[m]]
         
            results_array=results_array.T
            X, Y = np.meshgrid(X_vals, Y_vals)
            Z=results_array*factor#np.log10(abs(results_array))
            #ax.set_xlim([-10, -8])
            CS=ax.plot_surface(X,np.log10(Y),Z, cmap=cm.viridis_r)
            surfaces.append(CS)
            ax.set_title("{0} Hz".format(freqs[i]))
            #ax.xaxis.set_major_formatter(lambda x, pos:"$10^{%d}$"% x  )
            ax.yaxis.set_major_formatter(lambda x, pos:"$10^{%d}$"% x )
            ax.zaxis.set_major_formatter(ticker.PercentFormatter())
            #ax.text(-7.5, 2.5, 0, "$k_0$", color='red')
            #ax.text(-9.25, min(np.log10(Y_vals))-1.1, 0, r"$\Gamma$", color='red')
            #ax.set_ylabel("$k_0$"5
            #ax.set_xlabel(r"$\Gamma$")
            #ax.set_zlabel("Score")
            """if param1 in log_params:
                #ax.set_xscale("log")
                
            #ax.set_yscale("log")
            #
            if param_counter_1!=0:
                ax.set_yticks([])
          
            if param_counter_1==len(params)-1:
                twinx=ax.twinx()
                twinx.set_ylabel(dirlabels[j], rotation=0)
                twinx.set_yticks([])
            if (i*2)+j!=(len(freqs)*2)-1:
                ax.set_xticks([])
            else:
                all_titles[3]="$C_{dl}$"
                ax.set_xlabel(all_titles[param_counter_1])
            if i==0 and param_counter_1==0 and j==0:
                colorax = fig.add_axes([0.45,0.03,0.35,0.01])
                fig.colorbar(CS, cax=colorax, orientation='horizontal')
                colorax.set_xlim([0, 100])
                colorax.text(-0.55, 0.5, r"Normalised fraction of $\log|\mathcal{L}(\theta | I(t))|$",
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=colorax.transAxes)
               
                colorax.xaxis.set_major_formatter(ticker.PercentFormatter())"""
                

  

"""for i in range(0, len(params)):
    first_xlim=axes[0,i].get_xlim()
    consensus_xlim=first_xlim
    for j in range(1, len(freqs)*2):
        curr_xlim=axes[j,i].get_xlim()
        consensus_xlim=[max(curr_xlim[0], consensus_xlim[0]), min([curr_xlim[1], consensus_xlim[1]])]
    for j in range(0, len(freqs)*2):
        axes[j,i].set_xlim(*consensus_xlim)
"""
"""for j in range(0, len(freqs)):
        axes_list[j].set_ylim([1.25, 3.25])
        axes_list[j].set_yticks([2, 3])
        if j%7==0:
            axes_list[j].yaxis.set_major_formatter(lambda x, pos:"$10^{%d}$"% x if x>0 else "0" )
            #plt.setp(ax.get_yticklabels(), rotation=55, ha='right')
        else:
            axes_list[j].set_yticks([])"""
fig.set_size_inches(12, 5)
plt.subplots_adjust(top=0.968,
bottom=0.08,
left=0.05,
right=0.966,
hspace=0.2,
wspace=0.2)
plt.show()
frames=360
interval=20
elevation=30
anim = animation.FuncAnimation(
    fig, update,
    frames=frames,
    interval=interval,
    fargs=(axes_list, surfaces, elevation),
    blit=False
)

# Save animation
anim.save("Rotator_E0.gif", writer='pillow')
plt.close()

