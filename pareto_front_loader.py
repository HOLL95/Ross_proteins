import numpy as np
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.pareto_frontier import plot_pareto_frontier
import itertools
import time

time.sleep(7200)
ax_client=np.load("/home/henryll/Documents/Ross_protein/frontier_tests/frontier_test_2.npy", allow_pickle=True).item()["saved_frontier"]
#render(plot_pareto_frontier(loaded_frontier["saved_frontier"], CI_level=0.90))

objectives = ax_client.experiment.optimization_config.objective.objectives
all_metrics=[objectives[x].metric for x in range(0, len(objectives))]
all_freqs=[vars(x)["_name"] for x in all_metrics]
metric_dict=dict(zip(all_freqs, all_metrics))
combinations=list(itertools.combinations(all_freqs, 2))
for i in range(0, len(combinations)):
    freq1=combinations[i][0]
    freq2=combinations[i][1]
    frontier = compute_posterior_pareto_frontier(
        experiment=ax_client.experiment,
        #data=ax_client.experiment.fetch_data(),
        primary_objective=metric_dict[freq1],
        secondary_objective=metric_dict[freq2],
        absolute_metrics=[freq1, freq2],
        num_points=50,
    )
    
    np.save("frontier_tests/frontier_results/set2/{0}_{1}".format(freq1, freq2), {"frontier":frontier})