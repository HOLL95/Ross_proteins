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
from sklearn.metrics import pairwise_distances
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
    param_dict={} 
    for i in range(0, num_metrics):#row
        for j in range(0, num_metrics):#column
            
            if i>j:
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
                            
                            global_points.append([point["parameters"][x] for x in frontier.param_dicts[q].keys()])
                        
                        xaxis=[x["scores"][key2] for x in points]
                        yaxis=[y["scores"][key1] for y in points]
                        found_neg=False
global_points=np.array(global_points    )
clustering = cluster.DBSCAN(eps=0.05, min_samples=35).fit(global_points)
labels=clustering.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters)
print("Estimated number of noise points: %d" % n_noise)
centroids = {}
medoids = {}

# Calculate centroids and medoids for each cluster
for cluster_id in range(n_clusters):
    # Get points in this cluster
    cluster_points = global_points[labels == cluster_id]
    
    if len(cluster_points) == 0:
        continue
    
    # Calculate centroid (mean of points)
    centroid = np.mean(cluster_points, axis=0)
    centroids[cluster_id] = centroid
    
    # Calculate medoid (actual point with minimum distance to all other points)
    # First calculate pairwise distances between all points in the cluster
    distances = pairwise_distances(cluster_points)
    
    # Find the point with minimum sum of distances to all other points
    medoid_idx = np.argmin(np.sum(distances, axis=1))
    medoid = cluster_points[medoid_idx]
    medoids[cluster_id] = medoid

# Print results
print("\nCentroids:")
for cluster_id, centroid in centroids.items():
    print(f"Cluster {cluster_id}: {list(centroid)}")
print("\nMedoids:")
for cluster_id, medoid in medoids.items():
    print(f"Cluster {cluster_id}: {list(medoid)}")