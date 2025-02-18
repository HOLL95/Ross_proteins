import numpy as np
import Surface_confined_inference as sci
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
frequencies=[3,9,15]
file_names=["{0}_Hz_Full_table.txt".format(x) for x in frequencies]
f_titles=['$E^0 \\mu$ (V)', '$E^0 \\sigma$ (V)', '$k_0$ ($s^{-1}$)', '$\\Gamma$ (mol cm$^{-2}$)', '$\\alpha$']
c_titles=['$R_u$ ($\\Omega$)', '$C_{dl}$ (F)', '$C_{dlE1}$', '$C_{dlE2}$', '$C_{dlE3}$',]
m_titles=['$\\omega$ (Hz)','Dimensionless score']
index_dict={}
log_params=['$k_0$ ($s^{-1}$)', '$R_u$ ($\\Omega$)']
format_params=[ '$C_{dl}$ (F)', '$C_{dlE1}$', '$C_{dlE2}$', '$C_{dlE3}$']
all_titles=[f_titles, c_titles, m_titles]
boundaries={
'$k_0$ ($s^{-1}$)': [5, 5000], 
'$E^0 \\mu$ (V)': [-0.45, -0.37],
'$C_{dl}$ (F)': [1e-6, 5e-4],
'$\\Gamma$ (mol cm$^{-2}$)': [1e-11, 8e-10],
'$R_u$ ($\\Omega$)': [0.1, 4000],
'$E^0 \\sigma$ (V)':[0.025, 0.15],
'$C_{dlE1}$':[-1e-2, 1e-2],
'$C_{dlE2}$':[-5e-3, 4e-3],
'$C_{dlE3}$':[-5e-5, 5e-5],
'$\\alpha$':[0.4, 0.6],

}
xvals=[0.1, 0.11, 0.12]
markers=["o","^", "X"]
cmap = plt.get_cmap('viridis_r')

for i in range(0, len(all_titles)):
    current_list=all_titles[i]
    for j in range(0, len(all_titles[i])):
        index_dict[current_list[j]]=[i,j]
fig, axes=plt.subplots(4, 5)
for i in range(0, len(file_names)):
    values, titles=sci._utils.read_param_table(os.path.join("Fit_tables", file_names[i]),get_titles=True)
    best=np.array(values)
    scores=best[:,-1]
    idx=[range(0, 6)]#np.where(scores<(1.5*scores[0]))
    sizes=np.arange( 20-len(idx[0]),20)*2.5
    
    for j in range(0, len(best[0])):       
        ax=axes[*index_dict[titles[j]]]
        if titles[j] in m_titles:
            continue
        if "Gamma" in titles[j] or "C_{dl}" in titles[j]:
            area_factor=0.07/0.036
        
        else:
            area_factor=1
        scatterplt=ax.scatter([xvals[i]]*len(idx[0]), np.flip(best[idx, j][0]), color=cmap(list(np.linspace(0, 1, 6))), marker=markers[i])
        #if j==0:
        if titles[j]=='$C_{dl}$ (F)':
             ax.set_ylabel('$C_{dl}$ (F cm$^{-2}$)')
        else:
            ax.set_ylabel(titles[j])
        if titles[j] in log_params:
            ax.set_yscale("log")
        elif titles[j] in format_params:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        if titles[j] in boundaries:
            ax.set_ylim(*boundaries[titles[j]])
            #ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3,))
            #ax.set_yticks(boundaries[titles[j]])
        ax.set_xticks([])
for i in range(0,3):
    axes[0, 3].scatter(3, None,  marker=markers[i],  label="{0} Hz".format(frequencies[i]), color="black")
axes[0, 3].legend(ncols=3, loc="center", bbox_to_anchor=[1.5, 1.465], frameon=True)
colorax = fig.add_axes([0.09,0.94,0.45,0.03])
bounds = list(np.linspace(7, 1, 7))
upper=6
lower=1
N=6
deltac = (upper-lower)/(2*(N-1))
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
mapper=mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(np.linspace(lower-deltac,upper+deltac,6))
cb=fig.colorbar(mapper, cax=colorax, orientation='horizontal', spacing="proportional")
colorax.set_xticks([])
cb.ax.minorticks_off()
colorax.annotate("", xy=(0.35, -0.5), xytext=(0.65, -0.5), xycoords='axes fraction',
            arrowprops=dict(arrowstyle="<-"))
colorax.set_xlabel("Rank", labelpad=10)
likelihood_ax=[]
start=0.09
end=0.97
y0=0.215
y1=0.447283783783783786
sep=0.065
width=((end-start)-(sep*2))/3
height=y1-y0

for i in range(0, 5):
    axes[-2, i].set_axis_off()
    axes[-1, i].set_axis_off()
    
    if i<3:
        likelihood_ax.append(fig.add_axes([0.09+(width+sep)*i, y0, width, height]))

loc="Profile_likelihoods"
freqs=["3_Hz", "9_Hz","15_Hz"]
files=os.listdir(loc)
options=["log","identity"]
normalised=["normalised","raw"]

for p in range(0, 1):
    for z in range(0, 1):
        

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
            results_array=results_array.T
            X, Y = np.meshgrid(r_vals, k_vals)
            
            Z=results_array*factor#np.log10(abs(results_array))
            if i==2:
                CS=likelihood_ax[i].contourf(X*(255/185),Y,Z, 15,cmap=cm.viridis_r)
            else:
                CS=likelihood_ax[i].contourf(X,Y,Z, 15,cmap=cm.viridis_r)
            current_ytick=list(likelihood_ax[i].get_ylim())
            if i==0:
                existing_ytick=current_ytick
            else:
                if current_ytick[0]>existing_ytick[0]:
                    existing_ytick[0]=current_ytick[0]
           
            for axis in likelihood_ax:
                axis.set_ylim([52, current_ytick[1]])
            #divider = make_axes_locatable(likelihood_ax[i])
           
            if i==0:
                cax=fig.add_axes([0.09,0.05,1-(0.09+0.03),0.03])
                #cax = divider.append_axes('top', size='5%', pad=0.05)
                
                fig.colorbar(CS, cax=cax, orientation='horizontal')
                cax.xaxis.set_major_formatter(ticker.PercentFormatter())
                #cax.set_title(r"Normalised fraction of $\log|\mathcal{L}(\theta | I(t))|$")
                #cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)"""
                
            
            #likelihood_ax[i].axvline(90, color="black", linestyle="--")array
            #likelihood_ax[i].axvline(1000, color="black", linestyle="--")
            if i==2:
                likelihood_ax[i].set_xscale("log")
            likelihood_ax[i].set_yscale("log")
            if i==0:
                likelihood_ax[i].set_ylabel("$k^0$ ($s^{-1}$)")
            else:
                likelihood_ax[i].axvline(309, color="black", linestyle="--")
                #likelihood_ax[i].set_yticks([])
            #likelihood_ax[i].axhline(50, color="black", linestyle="--")
            likelihood_ax[i].axhline(123, color="black", linestyle="--")
            likelihood_ax[i].set_xlabel("$R_u$ ($\\Omega$)")
            split_title=freqs[i].split("_")
            likelihood_ax[i].set_title(" ".join(split_title))
            
for ax in likelihood_ax[:2]:
    if i==0:
        ax.xaxis.set_major_formatter(lambda x, pos:"$10^{%.2f}$"% np.log10(x) if x>0 else "0" )
    else:
        ax.xaxis.set_major_formatter(lambda x, pos:"$10^{%.1f}$"% np.log10(x) if x>0 else "0" )
    #plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
       

plt.subplots_adjust(
top=0.89,
bottom=0.185,
left=0.09,
right=0.97,
hspace=0.35,
wspace=0.81
)
fig.set_size_inches(10, 6.25)
init_x=axes[-1, 0].get_position()
final_x=axes[-1, -1].get_position()
print(init_x, final_x)
for i in range(0, 5):
    fig.align_ylabels(axes[:,i])
plt.show()
"""
for i in range(1, 4):
    axes[-1, -i].set_axis_off()

axes[1, -3].legend(ncols=3, loc="center", bbox_to_anchor=[0.5, -0.5], frameon=True)
plt.subplots_adjust(top=0.937,
bottom=0.081,
left=0.084,
right=0.988,
hspace=0.25,
wspace=0.693)

fig.set_size_inches(12.5, 4.5)
textx=0.83
texty=0.25



axes[-1, 1].set_ylabel("Dimensionless noise")
plt.show()
"""
fig.savefig("TopPlots.png", dpi=500)