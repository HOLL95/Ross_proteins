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

loc="/home/henryll/Documents/Experimental_data/Nat/joint"
#loc="/users/hll537/Experimental_data/jamie/set1"
loc="/users/hll537/Experimental_data/M4D2_joint"
files =os.listdir(loc)
run=int(sys.argv[1])
sw_freqs=[65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
experiments_dict={"FTACV":{"3_Hz":{}, "9_Hz":{},"15_Hz":{}, "21_Hz":{}}, "SWV":{}}
experiments_dict['FTACV']['3_Hz']['80']= {'E_start': np.float64(-0.8902481705989678), 'E_reverse': np.float64(-0.0005663458462876747), 'omega': np.float64(3.0347196895539086), 'phase': np.float64(6.059035018632212), 'delta_E': np.float64(0.07653233662675293), 'v': np.float64(0.05954572063421371)}
experiments_dict['FTACV']['15_Hz']['80']= {'E_start': np.float64(-0.8902605797349086), 'E_reverse': np.float64(-0.0005573884536691498), 'omega': np.float64(15.173689418589497), 'phase': np.float64(5.448960410224393), 'delta_E': np.float64(0.051806426739964634), 'v': np.float64(0.059547949680300125)}
experiments_dict['FTACV']['9_Hz']['80']= {'E_start': np.float64(-0.8902499779033282), 'E_reverse': np.float64(-0.0005430347722881201), 'omega': np.float64(9.10419726283959), 'phase': np.float64(5.692478143186857), 'delta_E': np.float64(0.06450753668694713), 'v': np.float64(0.059548694281080276)}
experiments_dict['FTACV']['21_Hz']['80']= {'E_start': np.float64(-0.8902642585586962), 'E_reverse': np.float64(-0.0005551049836591826), 'omega': np.float64(21.24318369655132), 'phase': np.float64(5.291741324611729), 'delta_E': np.float64(0.04201541965885772), 'v': np.float64(0.05954836661645596)}
experiments_dict['FTACV']['15_Hz']['280']= {'E_start': np.float64(-0.8900404672710482), 'E_reverse': np.float64(-0.0006392975440194792), 'omega': np.float64(15.173686024700986), 'phase': np.float64(5.440366237427825), 'delta_E': np.float64(0.17876387449314424), 'v': np.float64(0.05953022016638514)}
experiments_dict['FTACV']['3_Hz']['280']= {'E_start': np.float64(-0.8901199635261801), 'E_reverse': np.float64(-0.0006379235223668012), 'omega': np.float64(3.03471913390979), 'phase': np.float64(6.05269501468929), 'delta_E': np.float64(0.2672159297976847), 'v': np.float64(0.05953326889446427)}
experiments_dict['FTACV']['21_Hz']['280']= {'E_start': np.float64(-0.8900811295649749), 'E_reverse': np.float64(-0.000625431504669649), 'omega': np.float64(21.24318133865051), 'phase': np.float64(5.285841018363916), 'delta_E': np.float64(0.14497585403067986), 'v': np.float64(0.05953322299348709)}
experiments_dict['FTACV']['21_Hz']['280']= {'E_start': np.float64(-0.8900392326739417), 'E_reverse': np.float64(-0.0006217574272811), 'omega': np.float64(21.42689360258165), 'phase': np.float64(3.770088101350285), 'delta_E': np.float64(0.11428281275681479), 'v': np.float64(0.05953395752776987)}
experiments_dict['FTACV']['9_Hz']['280']= {'E_start': np.float64(-0.8900244598847762), 'E_reverse': np.float64(-0.0006520099910067856), 'omega': np.float64(9.104193706642272), 'phase': np.float64(5.682042161157082), 'delta_E': np.float64(0.22351633655321063), 'v': np.float64(0.059528212300390126)}
directions=["anodic","cathodic"]
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
for i in range(0, len(sw_freqs)):
    experiments_dict["SWV"]["{0}_Hz".format(sw_freqs[i])]={}
    for j in range(0, len(directions)):
        experiments_dict["SWV"]["{0}_Hz".format(sw_freqs[i])][directions[j]]={"omega":sw_freqs[i],
                                            "scan_increment":5e-3,#abs(pot[1]-pot[0]),
                                            "delta_E":0.8,
                                            "SW_amplitude":5e-3,
                                            "sampling_factor":200,
                                            "E_start":directions_dict[directions[j]]["E_start"],
                                            "v":directions_dict[directions[j]]["v"]}
    


FTV_parameters=["E0_mean","E0_std","k0","gamma","Ru", "Cdl","CdlE1","CdlE2","CdlE3","alpha",]
SWV_parameters=["E0_mean","E0_std","k0","gamma","alpha"]
class Experiment_evaluation:
    def __init__(self, parameters, dataloc, input_params):
        
        
        
        classes={}
        files=os.listdir(dataloc)
        self.all_keys=[]
        exps=list(input_params.keys())
        counter=0
        for i in range(0, len(exps)):
            experiment=exps[i]
            cond_1_list=list(input_params[experiment].keys())
            for j in range(0, len(cond_1_list)):
                cond_1=cond_1_list[j]
                cond_2_list=list(input_params[experiment][cond_1].keys())
                for k in range(0, len(cond_2_list)):
                   
                    cond_2=cond_2_list[k]
                    experiment_key="-".join([experiment, cond_1, cond_2])
                    self.all_keys.append(experiment_key)
                    print(experiment_key)
                    experiment_params=input_params[experiment][cond_1][cond_2]
                    extras={
                                        "Temp":278,
                                        "area":0.036,
                                        "N_elec":1,
                                        "Surface_coverage":1e-10}
                        
                    for key in extras:
                        experiment_params[key]=extras[key]
                    if experiment=="FTACV":
                        initexp="FTACV"
                    else:
                        initexp="SquareWave"
                    classes[experiment_key]={
                        "class":sci.RunSingleExperimentMCMC(initexp,
                                        experiment_params,
                                        problem="inverse",
                                        normalise_parameters=True
                                        )
                        } 
                    dummy_zero_class=sci.RunSingleExperimentMCMC(initexp,
                                        experiment_params,
                                        problem="forwards",
                                        normalise_parameters=False
                                        )
                    classes[experiment_key]["class"].boundaries={
                        "E0":[-0.6, -0.1],
                        "E0_mean":[-0.6, -0.1],
                        "E0_std":[1e-3, 0.1],
                        "k0":[0.1, 5e3],
                        "alpha":[0.4, 0.6],
                        "Ru":[0.1, 5e3],
                        "gamma":[1e-11, 1e-9],
                        "Cdl":[0,1e-3],
                        "CdlE1":[-5e-5, 5e-5],
                        "CdlE2":[-1e-5, 1e-5],
                        "CdlE3":[-1e-6, 1e-6],
                        "alpha":[0.4, 0.6]
                        }
                    strfreq=cond_1.split("_")[0]
                    #print(files, cond_1, experiment, cond_2)
                    file=[x for x in files if (cond_1 in x  and experiment in x and cond_2 in x)][0]
                    data=np.loadtxt(os.path.join(dataloc,file))
                    
                    if experiment=="FTACV":
                        dec_amount=16
                        time=data[::dec_amount,0]
                        current=data[::dec_amount,1]
                        classes[experiment_key]["data"]=classes[experiment_key]["class"].nondim_i(current)
                        classes[experiment_key]["times"]=classes[experiment_key]["class"].nondim_t(time)
                        classes[experiment_key]["class"].Fourier_fitting=True
                        classes[experiment_key]["class"].Fourier_window="hanning"
                        classes[experiment_key]["class"].top_hat_width=0.25
                        classes[experiment_key]["class"].Fourier_function="abs"
                        classes[experiment_key]["class"].Fourier_harmonics=list(range(3, 10))
                        classes[experiment_key]["FT"]=classes[experiment_key]["class"].experiment_top_hat(classes[experiment_key]["times"],  classes[experiment_key]["data"] )
                        #print(classes[experiment_key]["class"]._internal_memory["input_parameters"])
                        #plt.plot(classes[experiment_key]["FT"])
                        #plt.show()
                        classes[experiment_key]["class"].dispersion_bins=[30]
                        classes[experiment_key]["class"].optim_list=parameters["FTACV"]
                        classes[experiment_key]["data"]=classes[experiment_key]["class"].nondim_i(current)
                        classes[experiment_key]["times"]=classes[experiment_key]["class"].nondim_t(time)
                        dummy_zero_class.dispersion_bins=[1]
                        dummy_zero_class.optim_list=parameters["FTACV"]
                        worst_case=dummy_zero_class.simulate([-0.425, 0.1, 100, 1e-10,100, 1.8e-4,  1e-5, 1e-5, -1e-6,0.5], classes[experiment_key]["times"])             
                        ft_worst_case=classes[experiment_key]["class"].experiment_top_hat(classes[experiment_key]["times"], worst_case)
                        classes[experiment_key]["zero_point"]=sci._utils.RMSE(worst_case, classes[experiment_key]["data"])
                        classes[experiment_key]["zero_point_ft"]=sci._utils.RMSE(ft_worst_case, classes[experiment_key]["FT"])
                        #if cond_2=="80":
                        #    plt.plot(range(counter*len(classes[experiment_key]["data"]), (counter+1)*len(classes[experiment_key]["data"])),classes[experiment_key]["FT"]/(experiment_params["omega"]*experiment_params["delta_E"]))
                        #    counter+=1
                        #plt.plot(worst_case)
                        #plt.show()
                    else:
                        try:
                            data=np.loadtxt(os.path.join(loc, file), delimiter=",")
                        except:
                            data=np.loadtxt(os.path.join(loc, file))  
                        current=data[:-1,1]
                        times=classes[experiment_key]["class"].calculate_times()
                        voltage=classes[experiment_key]["class"].get_voltage(times)
                        pot=np.array([voltage[int(x)] for x in classes[experiment_key]["class"]._internal_memory["SW_params"]["b_idx"]])
                        #plt.plot(voltage)
                        #plt.plot(classes[experiment_key]["class"]._internal_memory["SW_params"]["b_idx"], pot)
                        #plt.plot(classes[experiment_key]["class"]._internal_memory["SW_params"]["b_idx"], data[:-1,0])
                        #plt.show()
                        classes[experiment_key]["class"].dispersion_bins=[30]
                        classes[experiment_key]["class"].fixed_parameters={"Cdl":0}
                        classes[experiment_key]["class"].optim_list=parameters["SWV"]
                        signal_region=[-0.425-0.15, -0.425+0.15]
                        signal_idx=np.where((pot>signal_region[0]) & (pot<signal_region[1]))
                        before=np.where((pot<signal_region[0]))
                        after=np.where((pot>signal_region[1]))
                        data=[]
                        noise_spacing=10
                        roll=20
                        midded_current=sci._utils.moving_avg(current, roll)
                        
                       
                        
                    
                       
                        for sequence in [pot, midded_current]:
                            catted_sequence=np.concatenate([sequence[before][roll+10-1::noise_spacing],sequence[after][roll+10-1::noise_spacing]])
                            data.append(catted_sequence)
                        #print(catted_sequence)
                        sortargs=np.argsort(data[0])
                        #plt.scatter([data[0][x] for x in sortargs], classes[experiment_key]["class"].nondim_i([data[1][x] for x in sortargs]), color="red")
                        CS=CubicSpline([data[0][x] for x in sortargs], [data[1][x] for x in sortargs])
                        worst_case=np.zeros(len(current))
                        classes[experiment_key]["data"]=classes[experiment_key]["class"].nondim_i(current-CS(pot))
                        classes[experiment_key]["times"]=classes[experiment_key]["class"].calculate_times()
                        #if cond_2=="cathodic":
                        #    plt.plot(range(counter*len(classes[experiment_key]["data"]), (counter+1)*len(classes[experiment_key]["data"])),classes[experiment_key]["data"]/(experiment_params["omega"]))
                        #    counter+=1
                        #plt.plot(pot, worst_case)
                        
                        #plt.plot(pot, classes[experiment_key]["data"])
                        #plt.plot(pot, classes[experiment_key]["class"].nondim_i(CS(pot)))
                        #plt.plot(pot, classes[experiment_key]["class"].nondim_i(current))
                        #plt.show()
                        
                        
                        
                        classes[experiment_key]["zero_point"]=sci._utils.RMSE(worst_case, classes[experiment_key]["data"])
                    del data    
                    
                    
                
                    classes[experiment_key]["class"].GH_quadrature=True
                   
                  
                  
                   
                self.classes=classes
                self.parameter_names=parameters
                self.input_parameters=input_params
            plt.show()
    def parse_input(self, parameters, key, cond_2):
        in_optimisation=False
        try:
            values=copy.deepcopy([parameters.get(x) for x in self.parameter_names["optimisation"]])
            in_optimisation=True
        except:
            pass
        if in_optimisation==True:
            if key=="FTACV":
                index="_1"
            else:
                index="_2"
            simvals=[]
            for param in self.parameter_names[key]:
                if param in self.parameter_names["optimisation"]:
                    simvals.append(parameters.get(param))
                else:
                    simvals.append(parameters.get(param+index))
        else:

            valuedict=dict(zip(self.parameter_names["optimisation"], copy.deepcopy(parameters)))
            
            if key=="FTACV":
                index="_1"
            else:
                index="_2"
            simvals=[]
            for param in self.parameter_names[key]:
                if param in self.parameter_names["optimisation"]:
                    if param=="E0_mean":
                        if cond_2=="anodic":
                            valuedict[param]+=valuedict["E0_mean_offset"]
                        elif cond_2=="cathodic":
                            valuedict[param]-=valuedict["E0_mean_offset"]
                    simvals.append(valuedict[param])
                else:
                    simvals.append(valuedict[param+index])
        
        return simvals
    
    def evaluate(self, parameters, group_dict):
        results_dict={}
        
        
        for key in self.classes.keys():
            experiment, cond_1, cond_2=key.split("-")
            freq=int(cond_1.split("_")[0])
            values=self.parse_input(parameters, experiment, cond_2)
            #values=[x for x in parameters]
            data=self.classes[key]["data"]
            
            omega=self.classes[key]["class"]._internal_memory["input_parameters"]["omega"]
            
            if experiment=="FTACV":
                sim=self.classes[key]["class"].simulate(values, self.classes[key]["times"])
                FT=self.classes[key]["class"].experiment_top_hat(self.classes[key]["times"], sim)
                FT_err=sci._utils.RMSE(FT, self.classes[key]["FT"])
                sim_err=sci._utils.RMSE(sim, data)
                for groupkey in group_dict.keys():
                    if group_dict[groupkey]["experiment"]=="FTACV":
                        
                        if group_dict[groupkey]["match"]==cond_2:
                            if group_dict[groupkey]["type"]=="ts":
                                deltaE=self.classes[key]["class"]._internal_memory["input_parameters"]["delta_E"]
                                if groupkey in results_dict:
                                    results_dict[groupkey]+=sim_err/(omega*deltaE)
                
                                else:
                                    results_dict[groupkey]=sim_err/((omega*deltaE))
                                
                            if group_dict[groupkey]["type"]=="ft":
                                if groupkey in results_dict:
                                    results_dict[groupkey]+=FT_err/(omega*deltaE)
                                else:
                                    results_dict[groupkey]=FT_err/(omega*deltaE)
                                
            elif experiment=="SWV":
                sim=self.classes[key]["class"].simulate(values, self.classes[key]["times"])
                sim_err=sci._utils.RMSE(sim, data)
                
                for groupkey in group_dict.keys():
                    if group_dict[groupkey]["experiment"]=="SWV":
                        right_exp=False
                        if "greater" in groupkey:
                            if int(group_dict[groupkey]["greater"])<freq:
                                right_exp=True
                        elif "lesser" in groupkey:
                            if int(group_dict[groupkey]["lesser"])>=freq:
                                right_exp=True
                                #plt.title(cond_1)
                                #plt.plot(sim)
                                #plt.plot(data)
                                #plt.show()
                        if right_exp==True:
                            if groupkey in results_dict:
                                results_dict[groupkey]+=sim_err/(omega)
                            else:
                                results_dict[groupkey]=sim_err/(omega)
                    
        return results_dict
param_dict={"SWV":SWV_parameters, "FTACV":FTV_parameters, "optimisation":["E0_mean", "E0_mean_offset","E0_std_1", "E0_std_2","k0","gamma_1", "gamma_2","Ru", "Cdl","CdlE1","CdlE2","CdlE3","alpha"]}

grouping_list=[{"experiment":"FTACV", "match":"80", "type":"ts"},
           {"experiment":"FTACV", "match":"280", "type":"ts"},
           {"experiment":"FTACV", "match":"80", "type":"ft"},
           {"experiment":"FTACV", "match":"280", "type":"ft"}, 
            {"experiment":"SWV", "lesser":"135", "type":"ts"}, 
            {"experiment":"SWV", "greater":"135", "type":"ts"},]
grouping_keys=[]
for grouping in grouping_list:
    if grouping["experiment"]=="FTACV":
        grouping_keys.append("-".join([grouping[x] for x in grouping.keys()]))
    else:
        if "lesser" in grouping.keys():
            grouping_keys.append("-".join([grouping["experiment"], "lesser", grouping["lesser"]]))
        else:
            grouping_keys.append("-".join([grouping["experiment"], "greater", grouping["greater"]]))

group_dict=dict(zip(grouping_keys, grouping_list))

simclass=Experiment_evaluation(param_dict, loc, experiments_dict)
ax_client = AxClient()
thresholds=simclass.evaluate([0.5 for x in param_dict["optimisation"]], group_dict)

param_arg=[
        {
            "name": param_dict["optimisation"][x],
            "type": "range",
            "value_type":"float",
            "bounds": [0.0, 1.0],
        }
        for x in range(0,len(param_dict["optimisation"]))
    ]
param_arg[1]["bounds"]=[0, 0.2]
objectives={}
print_tresh={}
for key in grouping_keys:

    objectives[key]=ObjectiveProperties(minimize=True, threshold=thresholds[key])
    


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
   
    ax_client.complete_trial(trial_index=trial_index, raw_data=simclass.evaluate(parameters, group_dict))
    

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
    #print("Saving")



