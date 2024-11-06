import numpy as np
import Surface_confined_inference as sci
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
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
markers=["o","^", "X"]
cmap = plt.get_cmap('viridis_r')
for i in range(0, len(all_titles)):
    current_list=all_titles[i]
    for j in range(0, len(all_titles[i])):
        index_dict[current_list[j]]=[i,j]
fig, axes=plt.subplots(3, 5)
for i in range(0, len(file_names)):
    values, titles=sci._utils.read_param_table(os.path.join("Fit_tables", file_names[i]),get_titles=True)
    print(titles)
    best=np.array(values)
    scores=best[:,-1]
    idx=[range(0, 6)]#np.where(scores<(1.5*scores[0]))
    sizes=np.arange( 20-len(idx[0]),20)*2.5
    
    for j in range(0, len(best[0])):       
        ax=axes[*index_dict[titles[j]]]
        scatterplt=ax.scatter([frequencies[i]]*len(idx[0]), np.flip(best[idx, j][0]), color=cmap(list(np.linspace(0, 1, 6))), marker=markers[i])
        #if j==0:
           
        ax.set_ylabel(titles[j])
        if titles[j] in log_params:
            ax.set_yscale("log")
        elif titles[j] in format_params:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
        if titles[j] in boundaries:
            ax.set_ylim(*boundaries[titles[j]])
        ax.set_xticks([])
for i in range(1, 4):
    axes[-1, -i].set_axis_off()
for i in range(0,3):
    axes[1, -3].scatter(3, None,  marker=markers[i],  label="{0} Hz".format(frequencies[i]), color="black")
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

plt.figtext(
    textx, texty, r"$E_r=E-IR_u$", horizontalalignment='center', fontsize=10, fontweight="bold"
)
plt.figtext(
    textx, texty-0.07, r"$I_c=C_{dl}(1+C_{dlE1}(E_r)+C_{dlE2}(E_r)^2+C_{dlE3}(E_r)^3)\frac{dEr}{dt}$",horizontalalignment='center', fontsize=10, fontweight="bold"
)
plt.figtext(
    textx, texty-0.07*2, r"$I_f=FA\Gamma(k_0(1-\theta) \exp((1-\alpha)(E_r-E^0))- k_0 (\theta)\exp(-\alpha(E_r-E^0)))$", horizontalalignment='center', fontsize=10, fontweight="bold"
)
colorax = fig.add_axes([0.425,0.13,0.225,0.05])
bounds = list(np.linspace(7, 1, 7))
upper=6
lower=1
N=6
deltac = (upper-lower)/(2*(N-1))
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
mapper=mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
mapper.set_array(np.linspace(lower-deltac,upper+deltac,6))
cb=fig.colorbar(mapper, cax=colorax, orientation='horizontal', spacing="proportional", ticks=[6.5, 5.5, 4.5, 3.5, 2.5, 1.5])
cb.ax.minorticks_off()
cb.ax.set_xticklabels([6, 5, 4, 3,2, 1])
colorax.set_xlabel("Rank")
axes[-1, 1].set_ylabel("Dimensionless noise")
plt.show()

fig.savefig("TopPlots.png", dpi=500)