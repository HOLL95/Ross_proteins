import numpy as np
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
import os
import Surface_confined_inference as sci
num_metrics=6
import itertools
import matplotlib.pyplot as plt
import time
from sklearn import cluster
def turn_metric_to_file(objective):
    string1=objective.metric_names[0]
    return "_".join(string1.split(":"))
def create_label(key):
    split_list=key.split("-")
    final_list=[split_list[x].split(":")[1] for x in range(0, 2)]
    final_list+=split_list[2:]
    return ":".join(final_list)
param_fig_created=False
global_points=[]
for m in range(0, 11):

    loc="/home/henryll/Documents/Frontier_results/M4D2_inference_4/set_{0}/".format(m)
    ax_client=np.load("{0}/ax_client.npy".format(loc), allow_pickle=True).item()["saved_frontier"]
    if m==0:
        objectives = ax_client.experiment.optimization_config.objective.objectives
        letters=["A","B","C","D","E","F"]
        objs=[x.metric_names[0] for x in objectives]
        obj_dict=dict(zip(objs, letters))
        num_metrics=len(objectives)
        fig, axes=plt.subplots(num_metrics, num_metrics)
    param_dict={} 
    for i in range(0, num_metrics):#row
        for j in range(0, num_metrics):#column
            
            if i>j:
                ax=axes[i,j]
                frontloc="{0}/fronts".format(loc)
                frontier_files=os.listdir(frontloc)
                metric1=turn_metric_to_file(objectives[i])
                metric2=turn_metric_to_file(objectives[j])
                key1=objectives[i].metric_names[0]

                key2=objectives[j].metric_names[0]
                for file in frontier_files:
                    
                    if metric1 in file and metric2 in file:
                        frontier=np.load(os.path.join(frontloc, file), allow_pickle=True).item()["frontier"]
                        #render(plot_pareto_frontier(loaded_frontier["saved_frontier"], CI_level=0.90))

                        points=[]
                        # For each point on the Pareto frontier
                        for q in range(len(frontier.param_dicts)):
                            point = {
                                "parameters": frontier.param_dicts[q],
                                "scores": {}
                            }
                            
                            # Add all metric scores for this point
                            for metric in frontier.means.keys():
                                point["scores"][metric] = frontier.means[metric][q]
                                
                            points.append(point)
                        
                        xaxis=[x["scores"][key2] for x in points]
                        yaxis=[y["scores"][key1] for y in points]
                        found_neg=False
                        if param_fig_created==False:
                            param_keys=list(points[0]["parameters"].keys())
                            row, col=sci._utils.det_subplots(len(param_keys))
                            param_fig, param_axes=plt.subplots(row, col)
                            param_fig_created=True
                           
                    
                        for key in param_keys:
                            value_list=[x["parameters"][key] for x in points]
                            if key in param_dict:
                                param_dict[key]+=value_list
                            else:
                                param_dict[key]=value_list
                        for array, key in zip([xaxis, yaxis], [key1, key2]):
                            check=[x<0 for x in array]
                            if any(check):
                                found_neg=True
                        if found_neg==False:
                            ax.scatter(xaxis, yaxis, c=sci._utils.colours[m%10])
                
                    if i==num_metrics-1:
                        ax.set_xlabel(obj_dict[key2])
                    if j==0:
                        ax.set_ylabel(obj_dict[key1])
            else:
                axes[i,j].set_axis_off()
    for z in range(0, len(param_keys)):
        pax=param_axes[z//col, z%col]
        pax.hist(param_dict[param_keys[z]], bins=20)
        pax.set_xlabel(param_keys[z])
for key in obj_dict.keys():
    axes[0, -1].scatter(0, 0, label="{0}:{1}".format(obj_dict[key],key), alpha=0)
axes[0,-1].legend()
plt.subplots_adjust(top=0.985,
bottom=0.06,
left=0.05,
right=0.992,
hspace=0.4,
wspace=0.4)
plt.show()
