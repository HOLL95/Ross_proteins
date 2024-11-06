import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import os
loc="/home/userfs/h/hll537/Documents/Experimental_data/SWV"
loc="/users/hll537/Experimental_data/SWV"
loc="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/SquareWave"
freqs=[25, 100, 150, 200, 250, 300, 350, 400, 450, 500]
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
files=os.listdir(loc)
paramloc="/home/henryll/Documents/Inference_results/swv/set4_9"
fig, axes=plt.subplots(5,4)
from matplotlib.patches import Rectangle
from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.ticker import MaxNLocator

plt.subplots_adjust(top=0.93,
bottom=0.05,
left=0.06,
right=0.975,
hspace=0.5,
wspace=0.485)
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
for i in range(0,len(freqs)):
    
    for j in range(0, len(directions)):
        
        idx=(i*2)+j
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
                                    square_wave_return="net",
                                    problem="forwards"

                                    )
        fw=sci.SingleSlurmSetup("SquareWave",
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
                                    square_wave_return="forwards",
                                    problem="forwards"

                                    )
        bw=sci.SingleSlurmSetup("SquareWave",
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
                                    problem="forwards"

                                    )
        times=sw_class.calculate_times()
        potential=sw_class.get_voltage(times)
        if idx%4==0:
            ax.set_ylabel("Current ($\\mu A$)", labelpad=7)
            
        if idx//4==4:
            ax.set_xlabel("Potential (V)", labelpad=7)
        else:
            ax.set_xticklabels([])
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
            bf_k_val=best_fit[2]
            k_val=[125, 300, bf_k_val]
            labels=[125, 300, "Best fit"]
            classes=[sw_class, fw, bw]
            for o in range(0, len(k_val)):
                param_values=copy.deepcopy(best_fit)
                param_values[2]=k_val[o]
                
                for k in range(0, len(classes)):
                    classes[k].dispersion_bins=[50]
                    classes[k].optim_list=["E0_mean","E0_std","k0","gamma","Cdl"]+extra_keys[:m]+["alpha"]
                    print(param_values, freqs[i])
                    sim=1e6*classes[k].dim_i(classes[k].Dimensionalsimulate(param_values[:-1], times))
                    if k==0:
                        linestyle="-"
                        label=labels[o]
                    else:
                        linestyle="--"
                        label=None
                    ax.plot(pot, sim, color=sci._utils.colours[o+1], label=label, linestyle=linestyle)
axes[0,1].legend(bbox_to_anchor=[1.15, 1.4], ncols=5, loc="center", frameon=False)
fig.set_size_inches(10, 12)
fig.savefig("SWV_all_sims.png", dpi=500)
plt.show()
            