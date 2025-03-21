from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import itertools
import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import CubicSpline

import os
from ax.utils.notebook.plotting import init_notebook_plotting, render
from pathlib import Path
from submitit import AutoExecutor
import sys
from draft_master_class import ExperimentEvaluation
run=int(sys.argv[1])
loc="/home/henryll/Documents/Experimental_data/Nat/joint"
loc="/users/hll537/Experimental_data/M4D2_joint"

sw_freqs=[65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
experiments_dict={}
dictionary_list=[
    {'E_start': np.float64(-0.8901199635261801), 'E_reverse': np.float64(-0.0006379235223668012), 'omega': np.float64(3.03471913390979), 'phase': np.float64(6.05269501468929), 'delta_E': np.float64(0.2672159297976847), 'v': np.float64(0.05953326889446427)},
    {'E_start': np.float64(-0.8900244598847762), 'E_reverse': np.float64(-0.0006520099910067856), 'omega': np.float64(9.104193706642272), 'phase': np.float64(5.682042161157082), 'delta_E': np.float64(0.22351633655321063), 'v': np.float64(0.059528212300390126)},
    {'E_start': np.float64(-0.8900404672710482), 'E_reverse': np.float64(-0.0006392975440194792), 'omega': np.float64(15.173686024700986), 'phase': np.float64(5.440366237427825), 'delta_E': np.float64(0.17876387449314424), 'v': np.float64(0.05953022016638514)},
]
FT_options=dict(Fourier_fitting=True,
                Fourier_window="hanning",
                top_hat_width=0.25,
                Fourier_function="abs",
                Fourier_harmonics=list(range(3, 10)), 
                dispersion_bins=[30],
                optim_list=["E0_mean","E0_std","k0","gamma","Ru", "Cdl","CdlE1","CdlE2","CdlE3","alpha"])
zero_ft=[-0.425, 0.1, 100, 8e-11,100, 1.8e-4,  1e-5, 1e-5, -1e-6,0.5]
zero_sw={"potential_window":[-0.425-0.15, -0.425+0.15], "thinning":10, "smoothing":20}
labels=["3_Hz", "9_Hz", "15_Hz"]
for i in range(0, len(labels)):
    experiments_dict=sci.construct_experimental_dictionary(experiments_dict, {**{"Parameters":dictionary_list[i]}, **{"Options":FT_options}, "Zero_params":zero_ft}, "FTACV", labels[i], "280_mV")

directions=["anodic","cathodic"]
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
sw_options=dict(square_wave_return="net", dispersion_bins=[30], optim_list=["E0_mean","E0_std","k0","gamma","alpha"])
for i in range(0, len(sw_freqs)):
    
    for j in range(0, len(directions)):
        params={"omega":sw_freqs[i],
        "scan_increment":5e-3,#abs(pot[1]-pot[0]),
        "delta_E":0.8,
        "SW_amplitude":5e-3,
        "sampling_factor":200,
        "E_start":directions_dict[directions[j]]["E_start"],
        "v":directions_dict[directions[j]]["v"]}
        experiments_dict=sci.construct_experimental_dictionary(experiments_dict, {**{"Parameters":params}, **{"Options":sw_options}, "Zero_params":zero_sw}, "SWV","{0}_Hz".format(sw_freqs[i]), directions[j])

bounds={
        "E0":[-0.6, -0.1],
        "E0_mean":[-0.6, -0.1],
        "E0_std":[1e-3, 0.08],
        "k0":[10, 500],
        "alpha":[0.4, 0.6],
        "Ru":[200, 400],
        "gamma":[1e-11, 1e-9],
        "Cdl":[0,1e-3],
        "CdlE1":[-5e-5, 5e-5],
        "CdlE2":[-1e-5, 1e-5],
        "CdlE3":[-1e-6, 1e-6],
        "alpha":[0.4, 0.6]
        }
common= {
        "Temp":278,
        "area":0.036,
        "N_elec":1,
        "Surface_coverage":1e-10}
evaluator=ExperimentEvaluation( loc, experiments_dict, bounds, common, SWV_e0_shift=True)
print(evaluator.all_parameters)
grouping_list=[
           {"experiment":"FTACV",  "type":"ts", "numeric":{"Hz":{"lesser":15}, "mV":{"equals":280}}, "scaling":{"divide":["omega", "delta_E"]}},
           {"experiment":"FTACV", "type":"ft", "numeric":{"Hz":{"lesser":15}, "mV":{"equals":280}}, "scaling":{"divide":["omega", "delta_E"]}}, 
            {"experiment":"FTACV", "type":"ts", "numeric":{"Hz":{"equals":15}, "mV":{"equals":280}}, "scaling":{"divide":["omega", "delta_E"]}},
           {"experiment":"FTACV",  "type":"ft", "numeric":{"Hz":{"equals":15}, "mV":{"equals":280}}, "scaling":{"divide":["omega", "delta_E"]}}, 
            {"experiment":"SWV", "numeric":{"Hz":{"lesser":135}}, "type":"ts", "scaling":{"divide":["omega"]}}, 
            #{"experiment":"SWV", "numeric":{"Hz":{"between":[135, 200]}}, "type":"ts", "scaling":{"divide":["omega"]}}, 
            {"experiment":"SWV", "numeric":{"Hz":{"geq":135}}, "type":"ts", "scaling":{"divide":["omega"]}}, ]


evaluator.initialise_grouping(grouping_list)

grouped_params={x:[range(0, 4), range(4, 6)] for x in ["E0_std","gamma"]}
evaluator.initialise_simulation_parameters(grouped_params)
#
thresholds=evaluator.get_zero_point_scores()
print(evaluator.all_parameters)
print(evaluator.parse_input([0.5 for x in evaluator.all_parameters]))
#evaluator.check_grouping()
#evaluator.results([0.5 for x in evaluator.all_parameters], target_key=evaluator.grouping_keys[0])
ax_client = AxClient()
param_arg=[
        {
            "name": x,
            "type": "range",
            "value_type":"float",
            "bounds": [0.0, 1.0],
        }
        if "offset" not in x else 
        
        {
            "name": x,
            "type": "range",
            "value_type":"float",
            "bounds": [0.0, 0.2],
        }
        
         for x in evaluator.all_parameters 
    ]

objectives={key:ObjectiveProperties(minimize=True, threshold=thresholds[key]) for key in evaluator.grouping_keys}

ax_client.create_experiment(
    name="Multi_experiment",
    parameters=param_arg,
    objectives=objectives,
    overwrite_existing_experiment=False,
    is_test=True,

)
paralell=ax_client.get_max_parallelism()
non_para_iterations=paralell[0][0]
directory=os.getcwd()
executor = AutoExecutor(folder=os.path.join(directory, "tmp_tests")) 
executor.update_parameters(timeout_min=60) # Timeout of the slurm job. Not including slurm scheduling delay.

executor.update_parameters(cpus_per_task=2)
executor.update_parameters(slurm_partition="nodes")
executor.update_parameters(slurm_job_name="mo_test")
executor.update_parameters(slurm_account="chem-electro-2024")
executor.update_parameters(mem_gb=2)
objectives = ax_client.experiment.optimization_config.objective.objectives
all_metrics=[objectives[x].metric for x in range(0, len(objectives))]
all_keys=[vars(x)["_name"] for x in all_metrics]
metric_dict=dict(zip(all_keys, all_metrics))
combinations=list(itertools.combinations(all_keys, 2))
print(combinations)
def save_current_front(input_dictionary):
    
    
    metrics=input_dictionary["metrics"]

    
    obj1=input_dictionary["combinations"][0]
    obj2=input_dictionary["combinations"][1]
    frontier = compute_posterior_pareto_frontier(
        experiment=input_dictionary["experiment"],
        #data=ax_client.experiment.fetch_data(),
        primary_objective=metrics[obj1],
        secondary_objective=metrics[obj2],
        absolute_metrics=[obj1, obj2],
        num_points=50,
    )

    np.save("frontier_results/set_{2}/fronts/{0}_{1}".format(obj1, obj2, input_dictionary["run"]), {"frontier":frontier})
Path(os.path.join(directory, "frontier_results","set_{0}".format(run), "fronts")).mkdir(parents=True, exist_ok=True)
    
for i in range(130):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
   
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluator.optimise_simple_score(parameters))
    

    #print("pre_saving")
    np.save("frontier_results/set_{1}/ax_client.npy".format(i, run), {"saved_frontier":ax_client})
    if i>non_para_iterations:
        #Path(os.path.join(directory, "frontier_results","set_{0}".format(run), "iteration_{0}".format(i))).mkdir(parents=True, exist_ok=True)
        with executor.batch():
            for j in range(0, len(combinations)):
                save_dict={}
                save_dict["metrics"]=metric_dict
                save_dict["combinations"]=combinations[j]
                save_dict["experiment"]=ax_client.experiment
                save_dict["run"]=run
                save_dict["iteration"]=i
                executor.submit(save_current_front, save_dict)
