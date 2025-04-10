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
from scipy.signal import decimate
import os
from ax.utils.notebook.plotting import init_notebook_plotting, render
from pathlib import Path
from submitit import AutoExecutor
import sys
from draft_master_class import ExperimentEvaluation

loc="/home/henryll/Documents/Experimental_data/Nat/joint"
#loc="/users/hll537/Experimental_data/M4D2_joint"

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

saved_sims=np.load("/home/henryll/Documents/Frontier_results/M4D2_inference_6/saved/simulation_values_b.npy", allow_pickle=True).item()
for key in list(saved_sims.keys())[2:]:
    target_keys=key.split("&")
    
    #evaluator.results([saved_sims[key][0]["saved_simulations"]], target_key=target_keys, savename=None, pre_saved=True, sim_plot_options="simple")
    evaluator.interactive_front_results(saved_sims[key], target_key=target_keys, sim_address="saved_simulations", score_address="score")

        

    
    
                
    
   
