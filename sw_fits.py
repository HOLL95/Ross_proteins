import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
file="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/SquareWave/SWV m4D2 PGE_2 mV_500 Hz_5 deg_cathodic.csv"
data=np.loadtxt(file, delimiter=",")
pot=data[:-1,0]
data_current=data[:-1,2]
input_dict={"omega":500,
                            "scan_increment":2e-3,#abs(pot[1]-pot[0]),
                            "delta_E":0.8,
                            "SW_amplitude":2e-3,
                            "sampling_factor":200,
                            "E_start":0,
                            "Temp":298,
                            "v":-1,
                            "area":0.07,
                            "N_elec":1,
                            "Surface_coverage":1e-10}
sw_class=sci.SingleExperiment("SquareWave",
                            input_dict,
                            square_wave_return="net",
                            problem="inverse"

                            )
times=sw_class.calculate_times()
potential=sw_class.get_voltage(times)
#plt.plot(pot, data_current)
#ax=plt.gca()
#twinx=ax.twinx()
#twinx.plot(pot, data[:-1, 2], color="red")
#plt.show()


num_steps=input_dict["delta_E"]/input_dict["scan_increment"]
potential_vals=np.zeros(len(potential))
potential_vals[0]=input_dict["E_start"]
plt.plot(times, potential)
"""for i in range(1, 2*int(num_steps)):

    
    if i%2==1:
        potential_vals[i]=potential_vals[i-1]+input_dict["v"]*(2*input_dict["SW_amplitude"])
    else:
        potential_vals[i]=potential_vals[i-1]-input_dict["v"]*(2*input_dict["SW_amplitude"])+(input_dict["v"]*input_dict["scan_increment"])

   
    plt.scatter(i*input_dict["sampling_factor"]//2, potential_vals[i], color="black")
plt.show()"""
print(len(sw_class._internal_memory["SW_params"]["b_idx"]), len(data_current))

#plt.scatter(sw_class._internal_memory["SW_params"]["b_idx"], pot, label="Data")
plt.scatter(sw_class._internal_memory["SW_params"]["b_idx"], sw_class._internal_memory["SW_params"]["E_p"], s=5, label="b_idx")
plt.scatter(sw_class._internal_memory["SW_params"]["f_idx"], sw_class._internal_memory["SW_params"]["E_p"]-input_dict["SW_amplitude"], s=5, color="red", label="f_idx")
plt.legend()
plt.show()

sw_class.boundaries={
"E0":[-0.5, -0.3],
"E0_mean":[-0.5, -0.3],
"E0_std":[1e-3, 0.1],
"k0":[0.1, 1e3],
"alpha":[0.4, 0.6],
"gamma":[1e-11, 1e-9],
"Cdl":[-10, 10],
"alpha":[0.4, 0.6],
"CdlE1":[-10, 10]
}
sw_class.dispersion_bins=[30]

sw_class.optim_list=["E0","k0","gamma","Cdl","alpha", "CdlE1"]
results=sw_class.Current_optimisation(sw_class._internal_memory["SW_params"]["b_idx"], sw_class.nondim_i(data_current), unchanged_iterations=200, tolerance=1e-8, dimensional=False, parallel=True)
print(results)
#results=[np.float64(-0.37662414409177947), np.float64(0.06914868425458533), np.float64(344.0590054855381), np.float64(3.29445437321673e-10), np.float64(0.028629232920096254), np.float64(0.5500069019648823), np.float64(0.0006478089511089012)]

bestfit=sw_class.dim_i(sw_class.simulate(results[:-1],times))
plt.plot(pot, data_current)
plt.plot(pot, bestfit)
plt.show()

