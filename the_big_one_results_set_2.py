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
files =os.listdir(loc)


sw_freqs=[65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
experiments_dict={"FTACV":{"3_Hz":{}, "9_Hz":{},"15_Hz":{}, "21_Hz":{}}, "SWV":{}}
#experiments_dict['FTACV']['3_Hz']['80']= {'E_start': np.float64(-0.8902481705989678), 'E_reverse': np.float64(-0.0005663458462876747), 'omega': np.float64(3.0347196895539086), 'phase': np.float64(6.059035018632212), 'delta_E': np.float64(0.07653233662675293), 'v': np.float64(0.05954572063421371)}
#experiments_dict['FTACV']['15_Hz']['80']= {'E_start': np.float64(-0.8902605797349086), 'E_reverse': np.float64(-0.0005573884536691498), 'omega': np.float64(15.173689418589497), 'phase': np.float64(5.448960410224393), 'delta_E': np.float64(0.051806426739964634), 'v': np.float64(0.059547949680300125)}
#experiments_dict['FTACV']['9_Hz']['80']= {'E_start': np.float64(-0.8902499779033282), 'E_reverse': np.float64(-0.0005430347722881201), 'omega': np.float64(9.10419726283959), 'phase': np.float64(5.692478143186857), 'delta_E': np.float64(0.06450753668694713), 'v': np.float64(0.059548694281080276)}
#experiments_dict['FTACV']['21_Hz']['80']= {'E_start': np.float64(-0.8902642585586962), 'E_reverse': np.float64(-0.0005551049836591826), 'omega': np.float64(21.24318369655132), 'phase': np.float64(5.291741324611729), 'delta_E': np.float64(0.04201541965885772), 'v': np.float64(0.05954836661645596)}
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
    def recover_parameters(self, parameters, experiment_key, norm):
        print(experiment_key)
        experiment, cond_1, cond_2=experiment_key.split("-")
        vals=self.parse_input(parameters, experiment, cond_2)
        if norm=="norm":
            return vals
        elif norm=="un_norm":
            return self.classes[experiment_key]["class"].change_normalisation_group(vals, "un_norm")
    
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
                            right_exp=False
                            if "greater" in groupkey:
                                if int(group_dict[groupkey]["greater"])<freq:
                                    right_exp=True
                            elif "lesser" in groupkey:
                                if int(group_dict[groupkey]["lesser"])>=freq:
                                    right_exp=True
                            if right_exp==True:
                                print(groupkey, key)
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
                        if right_exp==True:
                            if groupkey in results_dict:
                                results_dict[groupkey]+=sim_err/(omega)
                            else:
                                results_dict[groupkey]=sim_err/(omega)
                    
        return results_dict
    def results(self, parameters, group_dict, harmonics, mode, optimal_score=None, save=False):
        results_dict={}
        group_keys=list(group_dict.keys())
        subplots=sci._utils.det_subplots(len(group_keys))
        fig, ax=plt.subplots(*subplots)
        plot_dict={group_keys[i]:{"len":0, "axis":ax[i%subplots[0], i//subplots[0]], "maximum":0, "count":0} for i in range(0, len(group_keys))}
        ft_plot_dict={}
        
        # Highlight optimal score if provided
        for key in group_keys:
            if optimal_score==key:
                for axis in ['top','bottom','left','right']:
                    plot_dict[key]["axis"].spines[axis].set_linewidth(4)
                    plot_dict[key]["axis"].tick_params(width=4)
                    
        for key in self.classes.keys():
            experiment, cond_1, cond_2=key.split("-")
            freq=int(cond_1.split("_")[0])
            values=self.parse_input(parameters, experiment, cond_2)
            data=self.classes[key]["data"]
            omega=self.classes[key]["class"]._internal_memory["input_parameters"]["omega"]
            
            if experiment=="FTACV":
                sim=self.classes[key]["class"].simulate(values, self.classes[key]["times"])
                for groupkey in group_dict.keys():
                    plot_dict[groupkey]["axis"].set_title(groupkey)
                    if group_dict[groupkey]["experiment"]=="FTACV":
                        if group_dict[groupkey]["match"]==cond_2:
                            right_exp=False
                            if "greater" in groupkey:
                                if int(group_dict[groupkey]["greater"])<freq:
                                    right_exp=True
                            elif "lesser" in groupkey:
                                if int(group_dict[groupkey]["lesser"])>=freq:
                                    right_exp=True
                            if right_exp==True:
                                if group_dict[groupkey]["type"]=="ts":
                                    deltaE=self.classes[key]["class"]._internal_memory["input_parameters"]["delta_E"]
                                    xaxis=range(plot_dict[groupkey]["len"], plot_dict[groupkey]["len"]+len(data))
                                    data_norm=data/(omega*deltaE)
                                    sim_norm=sim/(omega*deltaE)
                                    if mode=="residual":
                                        plot_dict[groupkey]["axis"].plot(xaxis, data_norm-sim_norm)
                                    else:
                                        plot_dict[groupkey]["axis"].plot(xaxis, data_norm)
                                        plot_dict[groupkey]["axis"].plot(xaxis, sim_norm, linestyle="--", color="black", alpha=0.5)
                                    plot_dict[groupkey]["len"]+=len(data)

                                if group_dict[groupkey]["type"]=="ft":
                                    deltaE=self.classes[key]["class"]._internal_memory["input_parameters"]["delta_E"]
                                    data_harmonics=np.abs(sci.plot.generate_harmonics(self.classes[key]["times"], self.classes[key]["data"]/(deltaE*omega), hanning=True, one_sided=True, harmonics=harmonics))
                                    sim_harmonics=np.abs(sci.plot.generate_harmonics(self.classes[key]["times"], sim/(deltaE*omega), hanning=True, one_sided=True, harmonics=harmonics))
                                    ft_plot_dict[key]={"data":data_harmonics, "sim_harmonics":sim_harmonics}
                                    maximum=max(max(data_harmonics[0,:]), max(sim_harmonics[0,:]))
                                    plot_dict[groupkey]["maximum"]=max(maximum, plot_dict[groupkey]["maximum"])

            elif experiment=="SWV":
                sim=self.classes[key]["class"].simulate(values, self.classes[key]["times"])
                for groupkey in group_dict.keys():
                    if group_dict[groupkey]["experiment"]=="SWV":
                        right_exp=False
                        if "greater" in groupkey:
                            if int(group_dict[groupkey]["greater"])<freq:
                                right_exp=True
                        elif "lesser" in groupkey:
                            if int(group_dict[groupkey]["lesser"])>=freq:
                                right_exp=True
                        if right_exp==True:
                            xaxis=range(plot_dict[groupkey]["len"], plot_dict[groupkey]["len"]+len(data))
                            data_norm=data/omega
                            sim_norm=sim/omega
                            if mode=="residual":
                                plot_dict[groupkey]["axis"].plot(xaxis, data_norm-sim_norm)
                            else:
                                plot_dict[groupkey]["axis"].plot(xaxis, data_norm)
                                plot_dict[groupkey]["axis"].plot(xaxis, sim_norm, linestyle="--", color="black", alpha=0.5)
                            plot_dict[groupkey]["len"]+=len(data)

        # Plot FT harmonics
        for key in self.classes.keys():
            data=self.classes[key]["data"]
            experiment, cond_1, cond_2=key.split("-")
            freq=int(cond_1.split("_")[0])
            for groupkey in group_dict.keys():
                if group_dict[groupkey]["experiment"]=="FTACV":
                    if group_dict[groupkey]["match"]==cond_2:
                        right_exp=False
                        if "greater" in groupkey:
                            if int(group_dict[groupkey]["greater"])<freq:
                                right_exp=True
                        elif "lesser" in groupkey:
                            if int(group_dict[groupkey]["lesser"])>=freq:
                                right_exp=True
                        if right_exp==True and group_dict[groupkey]["type"]=="ft":
                            axis=plot_dict[groupkey]["axis"]
                            xaxis=range(plot_dict[groupkey]["len"], plot_dict[groupkey]["len"]+len(data))
                            datah=ft_plot_dict[key]["data"]
                            simh=ft_plot_dict[key]["sim_harmonics"]
                            
                            for i in range(0, len(harmonics)):
                                offset=(len(harmonics)-i)*1.1*plot_dict[groupkey]["maximum"]
                                ratio=plot_dict[groupkey]["maximum"]/(max(np.max(datah[i,:]), np.max(simh[i,:])))
                                datah[i,:]*=ratio
                                simh[i,:]*=ratio
                                if mode=="residual":
                                    axis.plot(xaxis, (datah[i,:]-simh[i,:])+offset, color=sci._utils.colours[plot_dict[groupkey]["count"]])
                                else:
                                    axis.plot(xaxis, datah[i,:]+offset, color=sci._utils.colours[plot_dict[groupkey]["count"]])
                                    axis.plot(xaxis, simh[i,:]+offset, color="black", linestyle="--")
                            plot_dict[groupkey]["count"]+=1
                            plot_dict[groupkey]["len"]+=len(data)

        # Set figure size and save if requested
        fig.set_size_inches(16,10)
        plt.tight_layout()
        #plt.show()
        if save==True:
            if optimal_score is not None:
                savefile=optimal_score+".png"
            else:
                savefile="results.png"
            fig.savefig(savefile, dpi=500)
        return fig, ax
    
param_dict={"SWV":SWV_parameters, "FTACV":FTV_parameters, "optimisation":["E0_mean", "E0_mean_offset","E0_std_1", "E0_std_2","k0","gamma_1", "gamma_2","Ru", "Cdl","CdlE1","CdlE2","CdlE3","alpha"]}

grouping_list=[
           {"experiment":"FTACV", "match":"280", "type":"ts", "greater":"9"},
           {"experiment":"FTACV", "match":"280", "type":"ft", "greater":"9"}, 
            {"experiment":"FTACV", "match":"280", "type":"ts", "lesser":"9"},
           {"experiment":"FTACV", "match":"280", "type":"ft", "lesser":"9"}, 
            {"experiment":"SWV", "lesser":"135", "type":"ts"}, 
            {"experiment":"SWV", "greater":"135", "type":"ts"},]
grouping_keys=[]
for grouping in grouping_list:

    grouping_keys.append("-".join(["{0}-{1}".format(x, grouping[x]) for x in grouping.keys()]))


group_dict=dict(zip(grouping_keys, grouping_list))
linestart=len(r"Parameterization:</em><br>")
save=False
if save==True:
    results={key:[] for key in grouping_keys}
    for m in range(0, 11):
        #files=os.listdir("/home/henryll/Documents/Frontier_results/M4D2_inference/set_{0}".format(m))
        
        
        
       
        ax_client=np.load("/home/henryll/Documents/Frontier_results/M4D2_inference/set_2_inf/set_{0}/ax_client.npy".format(m), allow_pickle=True).item()["saved_frontier"]
        print(ax_client.experiment.optimization_config.objective.objectives)
        #best_parameters= ax_client.get_pareto_optimal_parameters()
        #print(best_parameters)
        #frontier=load_pareto("/home/henryll/Documents/Jamie_protein/init_pareto_results/set_1/iteration_{0}/2.98_ts_2.98_ft.npy".format(iteration), parameters)

        for z in range(0, len(grouping_keys)):
            values=ax_client.get_contour_plot(param_x="k0", param_y="Cdl", metric_name=grouping_keys[z])
            arms=values[0]["data"][2]["text"]
            value_dict={}
            for i in range(0, len(arms)):
                key="arm{0}".format(i+1)
                first_split=arms[i].split("<br>")
                value_dict[key]={}
                value_dict[key]["score"]=float(first_split[1][first_split[1].index(":")+1:first_split[1].index("(")])
                
                for j in range(2, len(first_split)):
                    param_split=first_split[j].split(":")
                    value_dict[key][param_split[0]]=float(param_split[1])
                results[grouping_keys[z]].append(value_dict)
    np.save("init_pareto_results/save_points_set2.npy", results)
else:
    results=np.load("init_pareto_results/save_points_set2.npy", allow_pickle=True).item()
simclass=Experiment_evaluation(param_dict, loc, experiments_dict)
best_params={key:{} for key in grouping_keys}
all_scores={key:[] for key in grouping_keys}
experiment_keys=dict(zip(grouping_keys, [ "FTACV-3_Hz-280","FTACV-3_Hz-280","FTACV-3_Hz-280","FTACV-3_Hz-280", "SWV-135_Hz-anodic","SWV-135_Hz-cathodic"]))
keyr=list(results.keys())

for key in keyr:
    bestscore=1e6

    for i in range(0, len(results[key])):
        for arm in results[key][i].keys():
            all_scores[key].append(results[key][i][arm]["score"])
            if results[key][i][arm]["score"]<bestscore:
                bestscore=results[key][i][arm]["score"]
                best_params[key]["score"]=bestscore
                best_params[key]["params"]=[results[key][i][arm][key2] for key2 in param_dict["optimisation"]]
                best_params[key]["arm"]=arm
    print(key) 
    print(dict(zip(param_dict["optimisation"] , best_params[key]["params"])))
    #simclass.results(best_params[key]["params"], group_dict, list(range(3, 8)), "normal", key, save=True)
    #print(simclass.recover_parameters(best_params[key]["params"], experiment_keys[key], "un_norm"))
    #simclass.results(best_params[key]["params"], list(range(2, 10)))
figure, axis=plt.subplots(2,3)
listkey=list(results.keys())
for i in range(0, len(listkey)):
    ax1=axis[i%2, i//2]
    ax1.semilogy(sorted(all_scores[listkey[i]]))
    ax1.set_title(listkey[i])
plt.show()





#
