import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
file="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/SquareWave/SWV m4D2 PGE_2 mV_500 Hz_5 deg_anodic.csv"
data=np.loadtxt(file, delimiter=",")
pot=data[:-1,0]
data_current=data[:-1,1]
sw_class=sci.SingleExperiment("SquareWave",
                            {"omega":500,
                            "scan_increment":2e-3,#abs(pot[1]-pot[0]),
                            "delta_E":0.8,
                            "SW_amplitude":2e-3,
                            "sampling_factor":200,
                            "E_start":-0.8,
                            "Temp":298,
                            "v":1,
                            "area":0.07,
                            "N_elec":1,
                            "Surface_coverage":1e-10},
                            square_wave_return="backwards",
                            problem="inverse"

                            )
times=sw_class.calculate_times()
potential=sw_class.get_voltage(times)
plt.plot(pot, data_current)
ax=plt.gca()
twinx=ax.twinx()
twinx.plot(pot, data[:-1, 2], color="red")
plt.show()

plt.plot(times, potential)
print(len(sw_class._internal_memory["SW_params"]["b_idx"]), len(data_current))

plt.scatter(sw_class._internal_memory["SW_params"]["b_idx"], pot, label="Data")
plt.scatter(sw_class._internal_memory["SW_params"]["b_idx"], sw_class._internal_memory["SW_params"]["E_p"], s=5, label="b_idx")
plt.scatter(sw_class._internal_memory["SW_params"]["f_idx"], sw_class._internal_memory["SW_params"]["E_p"], s=5, color="red", label="f_idx")
plt.legend()
plt.show()

sw_class.boundaries={
"E0":[-0.5, -0.3],
"E0_mean":[-0.5, -0.3],
"E0_std":[1e-3, 0.1],
"k0":[0.1, 1e3],
"alpha":[0.4, 0.6],
"gamma":[1e-11, 1e-9],
"Cdl":[1e-3, 0.5],
"alpha":[0.4, 0.6]
}
sw_class.dispersion_bins=[30]

sw_class.optim_list=["E0","k0","gamma","Cdl","alpha"]
results=sw_class.Current_optimisation(sw_class._internal_memory["SW_params"]["b_idx"], sw_class.nondim_i(data_current), unchanged_iterations=200, tolerance=1e-8, dimensional=False, parallel=True)
print(results)
#results=[np.float64(-0.37662414409177947), np.float64(0.06914868425458533), np.float64(344.0590054855381), np.float64(3.29445437321673e-10), np.float64(0.028629232920096254), np.float64(0.5500069019648823), np.float64(0.0006478089511089012)]

bestfit=sw_class.dim_i(sw_class.simulate(results[:-1],times))
plt.plot(pot, data_current)
plt.plot(pot, bestfit)
plt.show()

