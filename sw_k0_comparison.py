import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
loc="/home/userfs/h/hll537/Documents/Experimental_data/SWV"
loc="/users/hll537/Experimental_data/SWV"
loc="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/SquareWave"
freqs=[25, 100, 150, 200, 250, 300, 350, 400, 450, 500]
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
files=os.listdir(loc)
paramloc="/home/henryll/Documents/Inference_results/swv/set4_4"
fig, axes=plt.subplots(6,4)
from matplotlib.patches import Rectangle
from matplotlib.transforms import TransformedBbox, Bbox
plt.subplots_adjust(top=0.93,
bottom=0.01,
left=0.075,
right=0.975,
hspace=0.5,
wspace=0.35)
def add_bounding_box(fig, axs,row, col_start, col_end, label, color, box_dims, textoffset):
    # Get the bounding box coordinates
   
    bbox1 = axs[row, col_start].get_position()
    bbox2 = axs[row, col_end].get_position()
    
    # Create rectangle coordinates
    x0 = bbox1.x0-box_dims[0]
    y0 = bbox2.y0-box_dims[1]
    width = bbox2.x1+2*box_dims[2] - bbox1.x0
    height = bbox1.y1+2*box_dims[3] - bbox2.y0
    
    # Add rectangle to the current axes
    ax = plt.gca()  # Get current axes
    rect = Rectangle(
        (x0, y0), width, height,
        fill=False,
        color=color,
        linestyle='-',
        linewidth=2,
        transform=fig.transFigure,
        clip_on=False
    )
    ax.add_patch(rect)
    
    # Add label at the top
    fig.text(
        x0 + width/2, bbox1.y1 + textoffset,
        label,
        ha='center',
        va='bottom',
        transform=fig.transFigure
    )
k0_values=[55.6962348699, 129.3318702126, 83.0799937618, 280.6447283944, 352.5144811136, 390.2623093351, 143.8441192437, 514.0955944754, 556.9630446625, 604.2001108766, 664.274577924, 761.0532114178, 389.5730197888, 899.3607035967, 843.1903107581, 894.1925888544, 520.4455440248, 999.9999998467, 338.4234345013, 999.9999999978]
import matplotlib
from matplotlib.ticker import MaxNLocator


cmap = plt.get_cmap('turbo')
colours=[cmap(x) for x in np.linspace(0.2, 1, len(k0_values))]
import matplotlib.patheffects as pe
for i in range(0,len(freqs)):
    
    for j in range(0, len(directions)):
        
        idx=(i*2)+j
        file=[x for x in files if (strfreqs[i] in x and directions[j] in x)][0]
        try:
            data=np.loadtxt(os.path.join(loc, file), delimiter=",")
        except:
            data=np.loadtxt(os.path.join(loc, file))
        pot=data[:-1,0]
        data_current=data[:-1,1]*1e6
        if j==0:
            label="Data"
        
        ax=axes[idx//4, idx%4]
        if j==0:
            if idx//4==4:
                bottom_stretch=0.012
            else:
                bottom_stretch=0
            print(bottom_stretch)
            add_bounding_box(fig, axes, idx//4,idx%4, idx%4+1, "{0} Hz".format(freqs[i]), "black", [0.028, 0.015+bottom_stretch, 0.02, 0.015+(bottom_stretch/2)], 0.015)
        
        ax.plot(pot,data_current, label=label)
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
                                    square_wave_return="backwards",
                                    problem="inverse"

                                    )
        times=sw_class.calculate_times()
        potential=sw_class.get_voltage(times)
        if idx%4==0:
            ax.set_ylabel("Current ($\\mu A$)", labelpad=5)
            
            
            
        if idx//4==4:
            ax.set_xlabel("Potential (V)")
        else:
            ax.set_xticklabels([])
        extra_keys=["CdlE1","CdlE2","CdlE3"]
        labels=[ "","Linear"]
       
        #newfilelist=file.split(".")
        #newfilelist[-1]="txt"
        #savefile=".".join(newfilelist)
        #np.savetxt(os.path.join(saveloc, savefile),np.column_stack((sw_class._internal_memory["SW_params"]["b_idx"], data_current, pot)),)
        for m in range(1, 2):
            sw_class.boundaries={
            "E0":[-0.5, -0.3],
            "E0_mean":[-0.5, -0.3],
            "E0_std":[1e-3, 0.1],
            "k0":[0.1, 1e3],
            "alpha":[0.4, 0.6],
            "gamma":[1e-11, 6e-10],
            "Cdl":[-10,10],
            "CdlE1":[-10, 10],
            "CdlE2":[-10, 10],
            "CdlE3":[-10, 10],
            "alpha":[0.4, 0.6]
            }
            sw_class.dispersion_bins=[50]
            #sw_class.fixed_parameters={"alpha":0.5}
            sw_class.optim_list=["E0_mean","E0_std","k0","gamma","Cdl"]+extra_keys[:m]+["alpha"]

            param_values=sci._utils.read_param_table(os.path.join(paramloc, "SWV_{0}".format(freqs[i]), directions[j], labels[m].lower(), "PooledResults_2024-09-26","Full_table.txt"))[0]
            orig_k=param_values[2]
            if idx<3:
                full_range=range(0, idx+3)
            elif idx>16:
                full_range=range(idx-3, len(k0_values))
            else:
                full_range=range(idx-3, idx+3)
            for q in range(0, len(full_range)):
                param_values[2]=k0_values[full_range[q]]
                
                sim=1e6*sw_class.dim_i(sw_class.Dimensionalsimulate(param_values[:-1], times))
                if param_values[2]==orig_k:
                    savesim=sim
                    savecol=colours[full_range[q]]
                else:
                    ax.plot(pot, sim, color=colours[full_range[q]], linestyle="-")
            ax.plot(pot, savesim, color=savecol, linestyle="--", path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=2))
            del savesim
    #axes[0,1].legend(bbox_to_anchor=[1.15, 1.4], ncols=5, loc="center", frameon=False)
for i in range(0, 4):
    axes[-1, i].set_axis_off()
for i in range(0, len(k0_values)):
    axes[-1, 2].plot(0,0, label="%.1f $s^{-1}$" % round(k0_values[i],1), color=colours[i])
axes[-1, 1].plot(0, 0, label="Data")
axes[-1,1].plot(0, 0, label="Fitted value", color="black", linestyle="--")
axes[-1, 1].legend(bbox_to_anchor=[-1, 0.5], loc="center", frameon=False)
axes[-1, 2].legend(ncols=5, bbox_to_anchor=[0, 0.5], loc="center", frameon=False)
fig.set_size_inches(10, 12)
fig.savefig("Comparative_sensitivity_k0.png", dpi=500)
plt.show()
            
