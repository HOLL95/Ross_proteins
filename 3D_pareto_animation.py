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
import matplotlib.pyplot as plt
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
parameters=['E0_mean', 'E0_std', 'k0', 'gamma', 'alpha']

#all_titles=sci._utils.get_titles(params+["k0"])
fig = plt.figure()
import matplotlib.pyplot as plt
log_parameters=["k0", "gamma"]#
axes_list=[]
surfaces=[]
def update(frame, axes, surfaces, elevation=30):
    """Update function for animation - rotates all surfaces."""
    for ax in axes:
        ax.view_init(elev=elevation, azim=frame)
    return axes
for m in range(0, len(freqs)):
    metric=str(freqs[m])
    #fig, axis=plt.subplots(len(parameters), len(parameters))



    ax = fig.add_subplot(2, 4, m+1, projection="3d")
    axes_list.append(ax)

    
    param2="k0"




    values=ax_client.get_contour_plot(param_x="k0", param_y="gamma", metric_name=metric)
    print(ax_client.experiment.optimization_config.objective.objectives)
    
    print(values.data["data"][1].keys())
    x_axis=values.data["data"][1]["x"]
    y_axis=values.data["data"][1]["y"]
    z=(np.array(values.data["data"][1]["z"]))
    unnormed_x=[sci._utils.un_normalise(x, boundaries["k0"]) for x in x_axis]
    unnormed_y=[sci._utils.un_normalise(x, boundaries["gamma"]) for x in y_axis]
    #if parameters[i] in log_parameters:
    #    axis[i,j].set_yscale("log")
    #    unnormed_y=np.log10(unnormed_y)
    #if parameters[j] in log_parameters:
    #axis[i,j].set_xscale("log")
    #    unnormed_x=np.log10(unnormed_x)
    min_score=np.min(z)
    max_score=np.max(z)
    for i in range(0, len(z)):
        for j in range(0, len(z[0])):
            z[i,j]=100*(1-sci._utils.normalise(z[i,j], [min_score, max_score]))
    X, Y = np.meshgrid(unnormed_x, unnormed_y)
    CS=ax.plot_surface(np.log10(X),np.log10(Y),np.array(z), cmap=cm.viridis_r)
    surfaces.append(CS)
    ax.set_yticks([ -10, -9])
    ax.xaxis.set_major_formatter(lambda x, pos:"$10^{%d}$"% x  )
    ax.yaxis.set_major_formatter(lambda x, pos:"$10^{%d}$"% x )
    
    plt.setp(ax.get_yticklabels(), rotation=360-35, ha='right')

    ax.zaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xlabel("$k_0$", labelpad=10)
    ax.set_ylabel(r"$\Gamma$", labelpad=10)
    ax.set_title("{0} Hz".format(freqs[m]))
    #ax.set_zlabel("Score")
    #ax.text(0.5, -0.25, 0, "$k_0$", color='red')
    #ax.text(1.25, 0.50, r"$\Gamma$", color='red')
    
fig.set_size_inches(12, 7)
plt.subplots_adjust(top=0.968,
bottom=0.08,
left=0.075,
right=0.956,
hspace=0.2,
wspace=0.2)         
#fig.set_size_inches(12, 12)

plt.show()
print("going")
frames=360
interval=50
elevation=30
anim = animation.FuncAnimation(
    fig, update,
    frames=frames,
    interval=interval,
    fargs=(axes_list, surfaces, elevation),
    blit=False
)

# Save animation
anim.save("pareto_rotator.gif", writer='pillow')
plt.close()
