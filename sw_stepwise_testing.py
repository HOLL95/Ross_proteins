import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import copy
input_dict={"omega":150,
                            "scan_increment":2e-3,#abs(pot[1]-pot[0]),
                            "delta_E":0.6,
                            "SW_amplitude":2e-3,
                            "sampling_factor":400,
                            "E_start":0,
                            "Temp":298,
                            "v":-1,
                            "area":0.07,
                            "N_elec":1,
                            "Surface_coverage":1e-10}
sim_class=sci.SWVStepwise(input_dict)
sw_class=sci.SingleExperiment("SquareWave", input_dict, square_wave_return="total")
sw_class.optim_list=["E0","k0","gamma","alpha", "Cdl"]
sim_class.optim_list=["E0","k0","Ru","gamma","alpha","Cdl","phase"]
values=[-0.2, 100, 100, 1e-10, 0.5, 1e-4, 0]
times=sim_class.calculate_times()
sw_potential=sim_class.get_voltage(times)
fig, ax=plt.subplots(1,4)
results=sim_class.simulate([-0.3, 100, 100, 1e-10, 0.5, 1e-4, 0], [])
data=[results["total"], results["theta"], sw_potential, sw_class.simulate([-0.2, 100, 1e-10, 0.5, 0], times)]
points= [sim_class._internal_memory["SW_params"]["b_idx"],sim_class._internal_memory["SW_params"]["f_idx"] ]
shapes=["o","^"]
for i in range(0, 4):
    ax[i].plot(data[i])
    for m in range(0, 2):
        xvals=points[m]
        y_vals=[data[i][int(x)] for x in points[m]]
    
        ax[i].scatter(xvals, y_vals, marker=shapes[m], color="black")


plt.show()

param_vals={"E0":[-0.5, -0.4, -0.3, -0.2],
            "k0":[1, 10, 100, 1000], 
            "Ru":[1, 10, 100, 1000],
            "Cdl":[1e-5, 4e-5, 1e-4]}
colours=sci._utils.colours
keys=list(param_vals.keys())
fig, axis=plt.subplots(2, 2)
for i in range(0, len(keys)):
    simvals=dict(zip(sim_class.optim_list, copy.deepcopy(values)))
    ax=axis[i//2, i%2]
    for j in range(0, len(param_vals[keys[i]])):
        simvals[keys[i]]=param_vals[keys[i]][j]
        results=sim_class.simulate([simvals[x] for x in sim_class.optim_list], [])


        ax.plot(results["E_p"], results["forwards"], label=param_vals[keys[i]][j], color=colours[j])
        ax.plot(results["E_p"], results["backwards"], linestyle="--", color=colours[j])
    ax.set_title(keys[i])
    ax.legend()
plt.show()
