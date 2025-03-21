import numpy as np
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
from kneed import KneeLocator
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

top_scores={'experiment:FTACV-type:ts-lesser:15Hz-equals:280mV': 11.1586, 
'experiment:FTACV-type:ft-lesser:15Hz-equals:280mV': 268.414, 
'experiment:FTACV-type:ts-equals:15Hz-equals:280mV': 3.77227, 
'experiment:FTACV-type:ft-equals:15Hz-equals:280mV': 46.2968, 
'experiment:SWV-type:ts-lesser:135Hz': 0.00467182, 
'experiment:SWV-type:ts-geq:135Hz': 0.00138112}
def normalise_score(point, key, top_score):
    score=point["scores"][key]
    ratio=(top_score[key]/score)*100
    return ratio

param_fig_created=False
global_points=[]
for m in range(0,11):
    fig_created=False

    loc="/home/henryll/Documents/Frontier_results/M4D2_inference_4/set_{0}/".format(m)
    ax_client=np.load("{0}/ax_client.npy".format(loc), allow_pickle=True).item()["saved_frontier"]
    if fig_created==False:
        objectives = ax_client.experiment.optimization_config.objective.objectives
        letters=["A","B","C","D","E","F"]
        objs=[x.metric_names[0] for x in objectives]
        obj_dict=dict(zip(objs, letters))
        num_metrics=len(objectives)
        fig, axes=plt.subplots(num_metrics, num_metrics)
        fig_created=True
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
                        print(frontier)
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
                        xaxis_bounds=[min(xaxis), max(xaxis)]
                        yaxis=[y["scores"][key1] for y in points]
                        yaxis_bounds=[min(yaxis), max(yaxis)]
                        #xaxis=[1-sci._utils.normalise(x, xaxis_bounds) for x in xaxis]
                        #yaxis=[1-sci._utils.normalise(x, yaxis_bounds) for x in yaxis]
                        found_neg=False
                        if param_fig_created==False:
                            param_keys=list(points[0]["parameters"].keys())
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
                            #for metric in frontier.means.keys():
                            ax.scatter(xaxis, yaxis, c=sci._utils.colours[m%10])
                            
                            k1=KneeLocator(xaxis, yaxis, curve="convex", direction="decreasing", online=False, S=1)
                            knee_x=k1.knee
                            knee_y=k1.knee_y
                            ax.scatter(knee_x, knee_y, s=40, edgecolors="black", label=1)
                            xidx=np.where(xaxis==knee_x)
                            yidx=np.where(yaxis==knee_y)
                            if len(xidx)>1 or len(yidx)>1:
                                common_idx=set(xidx[0]).intersection(set(y_idx[0]))[0]
                            else:
                                common_idx=xidx[0][0]
                            params=points[q]["parameters"]
                            
                            #ax.axvline(top_scores[key2], linestyle="--", color="black")
                            #ax.axhline(top_scores[key1], linestyle="--", color="black")
                
                    if i==num_metrics-1:
                        ax.set_xlabel(obj_dict[key2])
                    if j==0:
                        ax.set_ylabel(obj_dict[key1])
            else:
                axes[i,j].set_axis_off()
    for key in obj_dict.keys():
        axes[0, -1].scatter(0, 0, label="{0}:{1}".format(obj_dict[key],key), alpha=0)
    axes[0,-1].legend()
    axes[-1, 2].legend(loc="center", bbox_to_anchor=[0.5, 6])
    plt.subplots_adjust(top=0.985,
    bottom=0.06,
    left=0.05,
    right=0.992,
    hspace=0.4,
    wspace=0.4)
    plt.show()
