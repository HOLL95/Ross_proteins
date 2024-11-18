import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pints
import itertools
import sys
import copy
loc="/home/userfs/h/hll537/Documents/Experimental_data/SWV"
loc="/users/hll537/Experimental_data/SWV"
freqs=[25, 100, 150, 200, 250, 300, 350, 400, 450, 500]
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
files=os.listdir(loc)
i=int(sys.argv[1])//2
j=int(sys.argv[1])%2

file=[x for x in files if ("_{0}_".format(strfreqs[i]) in x and directions[j] in x and ".csv" in x)][0]
try:
    data=np.loadtxt(os.path.join(loc, file), delimiter=",")
except:
    data=np.loadtxt(os.path.join(loc, file))

pot=data[:-1, 0]
data_current=data[:-1,1]
sw_class=sci.RunSingleExperimentMCMC("SquareWave",
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
                            problem="inverse"

                            )
times=sw_class.calculate_times()
potential=sw_class.get_voltage(times)
netfile=np.savetxt(os.path.join("sw_net_data", file), np.column_stack((list(range(0, len(pot))), data_current, pot)))

extra_keys=["CdlE1","CdlE2","CdlE3"]
labels=["constant", "linear","quadratic","cubic"]
saveloc="/home/userfs/h/hll537/Documents/Experimental_data/SWV_slurm"
m=1
sw_class.boundaries={
"E0":[-0.5, -0.3],
"E0_mean":[-0.5, -0.3],
"E0_std":[0.02, 0.1],
"k0":[10, 1500],
"alpha":[0.35, 0.65],
"gamma":[5e-12, 1e-8],
"Cdl":[-1,1],
"CdlE1":[-1, 1],
"CdlE2":[-1, 1],
"CdlE3":[-1, 1],
}
sw_class.dispersion_bins=[50]
sw_class.num_cpu=25

sw_class.optim_list=["E0_mean","E0_std","k0","gamma","Cdl"]+extra_keys[:m]+["alpha"]
combinations=[[x]+["k0"] for x in sw_class.optim_list if "k0" not in x]
if directions[j]=="anodic":
 paramloc="inference_results_4_13/SWV_{0}/{1}/{2}/PooledResults_2024-11-14/Full_table.txt".format(freqs[i],directions[j],labels[m])
else:
 paramloc="inference_results_4_14/SWV_{0}/{1}/{2}/PooledResults_2024-11-15/Full_table.txt".format(freqs[i],directions[j],labels[m])
values=dict(zip(sw_class.optim_list, sci._utils.read_param_table(paramloc)[0][:-1]))

log_params=["gamma"]
combo_dict={}
for key in sw_class.optim_list:
    if key in log_params:
        combo_dict[key]=sci._utils.custom_logspace(sw_class.boundaries[key][0], sw_class.boundaries[key][1], values[key], 75)
    else:
        combo_dict[key]=sci._utils.custom_linspace(sw_class.boundaries[key][0], sw_class.boundaries[key][1], values[key], 75)
problem=pints.SingleOutputProblem(sw_class, list(range(0, len(pot))), sw_class.nondim_i(data_current))
likelihood=pints.SumOfSquaresError(problem)
print(combo_dict)


for o in range(0, len(combinations)):
    val1=combinations[o][0]
    val2=combinations[o][1]
    grid_values=list(itertools.product(combo_dict[val1], combo_dict[val2]))
    param_dict=copy.deepcopy(values)
    data_array=np.zeros(( len(grid_values), 3))
    for p in range(0, len(grid_values)):
    
        param_dict[val1]=grid_values[p][0]
        param_dict[val2]=grid_values[p][1]
        data_array[p,0]=param_dict[val1]
        data_array[p,1]=param_dict[val2]
        sim_values=[param_dict[x] for x in sw_class.optim_list]
        score=likelihood(sim_values)
        data_array[p,2]=score
        filename="tmp_scan_results/SWV_linear/scan-{0}-{1}-{2}-{3}".format(freqs[i],directions[j], val1, val2)
    np.savetxt(filename, data_array)
                
            
 
