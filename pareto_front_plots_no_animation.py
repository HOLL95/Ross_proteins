import numpy as np
import matplotlib.pyplot as plt
from ax.plot.pareto_frontier import plot_pareto_frontier
import os
import itertools
import Surface_confined_inference as sci
import matplotlib.ticker as ticker
loc="frontier_tests/frontier_results/set2"
import copy
files = os.listdir(loc)
linestart=len(r"Parameterization:</em><br>")
parameters=['E0_mean', 'E0_std', 'k0', 'gamma', 'alpha']
p_combinations=[('E0_mean', 'E0_std'), ('E0_mean', 'k0'), ('E0_mean', 'gamma'), ('E0_mean', 'alpha'), ('E0_std', 'k0'), ('E0_std', 'gamma'), ('E0_std', 'alpha'), ('k0', 'gamma'), ('k0', 'alpha'), ('gamma', 'alpha')]
freqs=[65, 75,85, 100, 115, 125, 135]
directions=["anodic","cathodic"]
colours=["cornflowerblue","darkorange","darkkhaki","red","magenta","mediumpurple", "seagreen"]
combinations=list(itertools.combinations(freqs,2))
strfreqs=[str(x) for x in freqs]
freqcolours=dict(zip(strfreqs, colours))
plot_colours=sci._utils.colours
ax_client=np.load("/home/henryll/Documents/Ross_protein/frontier_tests/frontier_test_2.npy", allow_pickle=True).item()["saved_frontier"]
thresholds={'65': 0.12264263913778613, '75': 0.12983951585847955, '85': 0.12785030588143828, '100': 0.14377940999664807, '115': np.float64(0.1322970338879717), '125': 0.1353898124271389, '135': 0.11645078613573955}
min_dict={str(x):[thresholds[str(x)],thresholds[str(x)]] for x in freqs}#
fig, axis=plt.subplots(2,7)
plot_dict={}
params_dict={}
boundaries={
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
    "alpha":[0.4, 0.6],
    "E0_offset":[0, 0.2],
    }
for i in range(0, len(combinations)):
    
    #ax=axis[i//3, i%3]
    freq1=str(combinations[i][0])
    freq2=str(combinations[i][1])
    key="{0}_{1}".format(freq1, freq2)
    plot_dict[key]={}
    params_dict[key]=[]
    file=[x for x in files if (freq1 in x and freq2 in x)][0]

    """filed=file[:file.index(".")]
    splitfile=filed.split("_")
    freq1=int(splitfile[0])
    freq2=int(splitfile[1])"""
    results_dict=np.load(os.path.join(loc, file), allow_pickle=True).item()["frontier"]
    values=plot_pareto_frontier(results_dict, CI_level=0.90)
    text=values.data["data"][0]["text"]

    x_scores=values.data["data"][0]["x"]
    y_scores=values.data["data"][0]["y"]
    min_dict[freq1][0]=min(min(x_scores), min_dict[freq1][0])
    min_dict[freq2][0]=min(min(y_scores), min_dict[freq2][0])
    x_err=values.data["data"][0]["error_x"]["array"]
    y_err=values.data["data"][0]["error_y"]["array"]
    plot_dict[key][freq1]=x_scores
    plot_dict[key][freq2]=y_scores
    text=values.data["data"][0]["text"]
    counter=0
    for line in text:
        starting_idx=line.find("Parameterization:")
        parameter_list=line[starting_idx+linestart:].split("<br>")
        param_dict={}
        for element in parameter_list:
            splitelement=element.split(":")
            param_dict[splitelement[0]]=float(splitelement[1])
        
        params_dict[key].append({"values":[param_dict[key] for key in param_dict.keys()],
                            "x":x_scores[counter],"y":y_scores[counter]})
        counter+=1
    #ax.errorbar(x_scores, y_scores, xerr=x_err, yerr=y_err)
    #ax.plot(x_scores, y_scores)
    #ax.set_xlabel(freq1+" Hz RMSE")
    #ax.set_ylabel(freq2+" Hz RMSE")
for i in range(0, len(freqs)):
    ax=axis[0, i]
    for j in range(0, len(freqs)):
        
        if freqs[i]==freqs[j]:
            continue
        else:
            for key in plot_dict.keys():
                if strfreqs[i] in key and strfreqs[j] in key:


        
            
                    raw_x=copy.deepcopy(plot_dict[key][strfreqs[i]])
                    raw_y=copy.deepcopy(plot_dict[key][strfreqs[j]])
                    #print(min_dict[freq1], freq1)
                    #print(min_dict[freq2], freq2)
                    
                    normed_x=[100*(1-sci._utils.normalise(x, min_dict[strfreqs[i]])) for x in raw_x]
                    normed_y=[100*(1-sci._utils.normalise(x, min_dict[strfreqs[j]])) for x in raw_y]
                    
                    
                    #print(min(raw_y), min(normed_y),min_dict[strfreqs[j]], np.array(raw_y)[np.where(raw_y<min_dict[strfreqs[j]])])
                    ax.plot(normed_x, normed_y, label=strfreqs[j], color=freqcolours[strfreqs[j]])
                    ax.set_xlabel(strfreqs[i]+" Hz Score")
                    ax.set_title(strfreqs[i]+" Hz")
                    if i ==0:
                        ax.set_ylabel("Score")
                    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
                    ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
                    
                    #ax.set_yscale("log")
                    #ax.set_xscale("log")
"""axis[-1, -1].set_axis_off()
for i in range(0, len(freqs)):
    axis[-1, -1].plot(0, 0, label=strfreqs[i]+" Hz", color=freqcolours[strfreqs[i]])
axis[-1,-1].legend()
"""

loc="/home/henryll/Documents/Experimental_data/Nat/m4d2_SET3"
files =os.listdir(loc)
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
        self.data_dict={}
        for i in range(0, len(frequencies)):
            self.data_dict[str(frequencies[i])]={"anodic":{},"cathodic":{}}
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
                classes[experiment_key]["data"]=data[:-1,  1]
                classes[experiment_key]["times"]=classes[experiment_key]["class"].calculate_times()
                self.data_dict[str(frequencies[i])][directions[j]]["current"]=data[:-1,  1]
                self.data_dict[str(frequencies[i])][directions[j]]["potential"]=data[:-1,  0]
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
            values=copy.deepcopy(parameters)#[parameters.get(x) for x in self.parameter_names]+[parameters.get("E0_offset")]
            data=self.classes[key]["data"]
            #print(key)
            if "anodic" in key:
                values[0]+=values[-1]
            #print(self.classes[key]["class"].optim_list, values)
            sim=self.classes[key]["class"].dim_i(self.classes[key]["class"].simulate(values[:-1], self.classes[key]["times"]))
            
            full_dict[key]=sim
        for i in range(0, len(self.freqs)):
            str_freq=str(self.freqs[i])
            return_dict[str_freq]={"anodic":full_dict[str_freq+"_anodic"], "cathodic":full_dict[str_freq+"_cathodic"]}
        return return_dict

simclass=SWV_evaluation(freqs, params, paramloc, loc)
step=10
savedict={}
simdict=np.load("frontier_tests/frontier_simulations/set2_savedsims.npy", allow_pickle=True).item()
simline={}
fig.set_size_inches(15,8)
plt.subplots_adjust(top=0.965,
bottom=0.097,
left=0.043,
right=0.982,
hspace=0.266,
wspace=0.432)
for i in range(0, len(combinations)):
    
    freq1=str(combinations[i][0])
    freq2=str(combinations[i][1])
    key="{0}_{1}".format(freq1, freq2)
    savedict[key]={}
    
    for j in range(0, len(params_dict[key]), step):

        get_params=params_dict[key][j]["values"]
        pareto_x=100*(1-sci._utils.normalise(params_dict[key][j]["x"], min_dict[freq1]))
        pareto_y=100*(1-sci._utils.normalise(params_dict[key][j]["y"], min_dict[freq2]))
        print(get_params)
        simulations=simclass.evaluate(get_params)
        for m in range(0, len(strfreqs)):
            savedict[key][strfreqs[m]]={}
            simline[strfreqs[m]]={}
            for q in range(0, len(directions)):
                print(key, strfreqs[m], directions[q])
                savedict[key][strfreqs[m]][directions[q]]=simulations[strfreqs[m]][directions[q]]
                if i==0:
                    axis[1,m].plot(simclass.data_dict[strfreqs[m]][directions[q]]["potential"],simclass.data_dict[strfreqs[m]][directions[q]]["current"], color=colours[m])

                    simline[strfreqs[m]][directions[q]],=axis[1,m].plot(simclass.data_dict[strfreqs[m]][directions[q]]["potential"],simdict[key][strfreqs[m]][directions[q]], color="black",linestyle="--")
                else:
                    simline[strfreqs[m]][directions[q]].set_ydata(simdict[key][strfreqs[m]][directions[q]])
                axis[1,m].set_xlabel("Potential (V)")
                if m==0:
                    axis[1,m].set_ylabel("Current (A)")
        if j==1:
            plt.show()
        
                
        np.save("frontier_tests/frontier_simulations/set2_savedsims_2", savedict)


