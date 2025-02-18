import numpy as np
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
import itertools
import time
import Surface_confined_inference as sci
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
from matplotlib import animation
import matplotlib.ticker as ticker

ax_client=np.load("/home/henryll/Documents/Ross_protein/frontier_tests/frontier_test_2.npy", allow_pickle=True).item()["saved_frontier"]
#render(plot_pareto_frontier(loaded_frontier["saved_frontier"], CI_level=0.90))
freqs=[65, 75,85, 100, 115, 125, 135]
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
all_titles=sci._utils.get_titles(params+["k0"])
fig = plt.figure()
import matplotlib.pyplot as plt
parameters=['E0_mean', 'E0_std', 'k0', 'gamma', 'alpha']
log_parameters=["k0", "gamma"]#
for m in range(0, len(freqs)):
    metric=str(freqs[m])
    fig, axis=plt.subplots(len(parameters), len(parameters))
    for i in range(0, len(parameters)):
        for j in range(0,len(parameters)):
            if j>=i:
                axis[i,j].set_axis_off()
            else:



                values=ax_client.get_contour_plot(param_x=parameters[j], param_y=parameters[i], metric_name=metric)
                print(ax_client.experiment.optimization_config.objective.objectives)
                break
                print(values.data["data"][1].keys())
                x_axis=values.data["data"][1]["x"]
                y_axis=values.data["data"][1]["y"]
                z=values.data["data"][1]["z"]
                unnormed_x=[sci._utils.un_normalise(x, boundaries[parameters[j]]) for x in x_axis]
                unnormed_y=[sci._utils.un_normalise(x, boundaries[parameters[i]]) for x in y_axis]
                if parameters[i] in log_parameters:
                    axis[i,j].set_yscale("log")
                #    unnormed_y=np.log10(unnormed_y)
                if parameters[j] in log_parameters:
                    axis[i,j].set_xscale("log")
                #    unnormed_x=np.log10(unnormed_x)
                X, Y = np.meshgrid(unnormed_x, unnormed_y)
                CS=axis[i,j].contourf(X,Y,z, 15)
                if i==len(parameters)-1:
                    axis[i, j].set_xlabel(parameters[j])
                if j==0:
                    axis[i, j].set_ylabel(parameters[i])
               
    fig.set_size_inches(12, 12)
    plt.tight_layout()
    #plt.show()
    fig.savefig("/home/henryll/Documents/Ross_protein/frontier_tests/frontier_surfaces/{0}.png".format(metric))