from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import os
from ax.utils.notebook.plotting import init_notebook_plotting, render
loc="/home/henryll/Documents/Experimental_data/Nat/m4d2_SET3"
files =os.listdir(loc)
freqs=[65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
strfreqs=[str(x) for x in freqs]

paramloc="/home/henryll/Documents/Inference_results/swv/set6_1"


params=["E0_mean","E0_std","k0","gamma", "alpha"]
line_params=["Cdl","CdlE1","CdlE2","CdlE3"]
class SWV_evaluation:
    def __init__(self, frequencies, parameters, paramloc, dataloc):
        original_params=["E0_mean","E0_std","k0","gamma", "Cdl","alpha","CdlE1","CdlE2","CdlE3"]
        directions=["anodic","cathodic"]
        directions_map=dict(zip(directions, ["1.csv", "2.csv"]))
        directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
        fixed_params=list(set(original_params)-set(parameters))
        classes={}
        files=os.listdir(dataloc)
        for i in range(0, len(frequencies)):
            
            for j in range(0, len(directions)):
                best_fit=sci._utils.read_param_table(os.path.join(paramloc, "SWV_{0}".format(freqs[i]), directions[j], "cubic", "PooledResults_2024-11-18","Full_table.txt"))[0]
                experiment_key="{0}_{1}".format(freqs[i], directions[j])
                input_params={"omega":freqs[i],
                                    "scan_increment":5e-3,#abs(pot[1]-pot[0]),
                                    "delta_E":0.8,
                                    "SW_amplitude":5e-3,
                                    "sampling_factor":200,
                                    "E_start":directions_dict[directions[j]]["E_start"],
                                    "Temp":278,
                                    "v":directions_dict[directions[j]]["v"],
                                    "area":0.036,
                                    "N_elec":1,
                                    "Surface_coverage":1e-10}
                classes[experiment_key]={
                    "class":sci.RunSingleExperimentMCMC("SquareWave",
                                    input_params,
                                    square_wave_return="net",
                                    problem="inverse",
                                    normalise_parameters=True
                                    )
                    } 
                dummy_zero_class=sci.RunSingleExperimentMCMC("SquareWave",
                                    input_params,
                                    square_wave_return="net",
                                    problem="inverse",
                                    normalise_parameters=True
                                    )
                classes[experiment_key]["class"].boundaries={
                    "E0":[-0.5, -0.3],
                    "E0_mean":[-0.5, -0.3],
                    "E0_std":[1e-3, 0.1],
                    "k0":[0.1, 5e3],
                    "alpha":[0.4, 0.6],
                    "gamma":[1e-11, 1e-9],
                    "Cdl":[-10,10],
                    "CdlE1":[-10, 10],
                    "CdlE2":[-10, 10],
                    "CdlE3":[-10, 10],
                    "alpha":[0.4, 0.6]
                    }
                best_fit=dict(zip(original_params, best_fit[:-1]))           
                file=[x for x in files if ("_{0}_".format(freqs[i]) in x and directions_map[directions[j]] in x)][0]
                
                try:
                    data=np.loadtxt(os.path.join(loc, file), delimiter=",")
                except:
                    data=np.loadtxt(os.path.join(loc, file))  
                classes[experiment_key]["class"].dispersion_bins=[30]
                classes[experiment_key]["class"].fixed_parameters={x:best_fit[x] for x in fixed_params}
                classes[experiment_key]["class"].optim_list=parameters
                classes[experiment_key]["data"]=classes[experiment_key]["class"].nondim_i(data[:-1,  1])
                classes[experiment_key]["times"]=classes[experiment_key]["class"].calculate_times()
                zero_dict={x:best_fit[x] for x in original_params }
                zero_dict["gamma"]=0
                dummy_zero_class.dispersion_bins=[1]
                dummy_zero_class.fixed_parameters=zero_dict
                dummy_zero_class.optim_list=[]
                worst_case=dummy_zero_class.simulate([], classes[experiment_key]["times"])
                
                classes[experiment_key]["zero_point"]=sci._utils.RMSE(worst_case, classes[experiment_key]["data"])
        self.classes=classes
        self.parameter_names=parameters
        self.freqs=frequencies
    def evaluate(self, parameters):
        
        
        full_dict={}
        return_dict={}
        for key in self.classes.keys():
            values=[parameters.get(x) for x in self.parameter_names]+[parameters.get("E0_offset")]
            data=self.classes[key]["data"]
            #print(key)
            if "anodic" in key:
                values[0]+=values[-1]
            #print(self.classes[key]["class"].optim_list, values)
            sim=self.classes[key]["class"].simulate(values[:-1], self.classes[key]["times"])
            
            full_dict[key]=sci._utils.RMSE(sim,data)
        for i in range(0, len(self.freqs)):
            str_freq=str(self.freqs[i])
            return_dict[str_freq]=full_dict[str_freq+"_anodic"]+full_dict[str_freq+"_cathodic"]
        return return_dict
                
simclass=SWV_evaluation(freqs, params, paramloc, loc)
ax_client = AxClient()

#
param_arg=[
        {
            "name": params[x],
            "type": "range",
            "value_type":"float",
            "bounds": [0.0, 1.0],
        }
        for x in range(0,len(params))
    ]
param_arg+=[{"name":"E0_offset", "type":"range","bounds":[0, 0.5],"value_type":"float"}]
objectives={}
print_tresh={}
for i in range(0, len(freqs)):
    str_freq=str(freqs[i])
    thresh=simclass.classes[str_freq+"_anodic"]["zero_point"]+simclass.classes[str_freq+"_cathodic"]["zero_point"]

    print_tresh[str_freq]=thresh
    objectives[str_freq]=ObjectiveProperties(minimize=True, threshold=thresh)
print(print_tresh)

ax_client.create_experiment(
    name="SWV_experiment",
    parameters=param_arg,
    objectives=objectives,
    overwrite_existing_experiment=True,
    is_test=True,

)

import time
import numpy as np
try_no=3
completed=False

    
for i in range(100):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=simclass.evaluate(parameters))
    

    print("pre_saving")
    np.save("frontier_tests/frontier_test_{0}.npy".format(try_no), {"saved_frontier":ax_client})
    print("Saving")



