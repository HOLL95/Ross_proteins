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

for m in range(0, 11):

    loc="/home/henryll/Documents/Frontier_results/M4D2_inference_4/set_{0}/".format(m)
    ax_client=np.load("{0}/ax_client.npy".format(loc), allow_pickle=True).item()["saved_frontier"]
    if m==0:
        objectives = ax_client.experiment.optimization_config.objective.objectives
        letters=["A","B","C","D","E","F"]
        objs=[x.metric_names[0] for x in objectives]
        obj_dict=dict(zip(objs, letters))
        num_metrics=len(objectives)
    param_dict={} 
    frontloc="{0}/fronts".format(loc)
    frontier_files=os.listdir(frontloc)

    for file in frontier_files:
        
        
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
        
        
        found_neg=False
        if param_fig_created==False:
            param_keys=list(points[0]["parameters"].keys())
            fig, ax=plt.subplots(len(param_keys), len(param_keys))
            param_fig_created=True
        for i in range(0, len(param_keys)):
            for j in range(0, len(param_keys)):
                if i>j:
                    xkey=param_keys[j]
                    ykey=param_keys[i]
                    xaxis=[]
                    yaxis=[]
                    for z in range(0, len(points)):
                        check_neg=all([points[z]["scores"][x]>0 for x in frontier.means.keys()])
                        if check_neg==True:
                            xaxis+=[points[z]["parameters"][xkey]]
                            yaxis+=[points[z]["parameters"][ykey]]
                    ax[i,j].scatter(xaxis, yaxis, c=sci._utils.colours[m%10], s=1)    
                    if i==len(param_keys)-1:
                        ax[i,j].set_xlabel(xkey)
                    else:
                        ax[i,j].set_xticks([])
                    if j==0:
                        ax[i,j].set_ylabel(ykey)
                    else:
                        ax[i,j].set_yticks([])
                            
                else:
                    ax[i,j].set_axis_off()
    

   

   
fig.set_size_inches(16, 10)
plt.subplots_adjust(top=1.0,
bottom=0.08,
left=0.04,
right=1.0,
hspace=0.2,
wspace=0.2)
#plt.show()
fig.savefig("pareto_test.png", dpi=250)
