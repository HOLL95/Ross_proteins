import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
file="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/SquareWave/SWV_m4D2_PGE_2_mV_500_Hz_5_deg_cathodic.csv"
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
                            "v":1,
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


num_steps=input_dict["delta_E"]/input_dict["scan_increment"]
potential_vals=np.zeros(len(potential))
potential_vals[0]=input_dict["E_start"]
def logistic(k, x, a):
    return 1/(1+a*np.exp(-2*k*x))
x=np.linspace(-5, 5, 1000)
for a in range(1, 10):
    plt.plot(x, logistic(2, x, a))
plt.show()
plt.plot(times, potential)

"""for i in range(1, 2*int(num_steps)):

    
    if i%2==1:
        potential_vals[i]=potential_vals[i-1]+input_dict["v"]*(2*input_dict["SW_amplitude"])
    else:
        potential_vals[i]=potential_vals[i-1]-input_dict["v"]*(2*input_dict["SW_amplitude"])+(input_dict["v"]*input_dict["scan_increment"])

   
    plt.scatter(i*input_dict["sampling_factor"]//2, potential_vals[i], color="black")
plt.show()"""
print(len(sw_class._internal_memory["SW_params"]["b_idx"]), len(data_current))

plt.scatter(sw_class._internal_memory["SW_params"]["b_idx"], pot, label="Data")
plt.scatter(sw_class._internal_memory["SW_params"]["b_idx"], sw_class._internal_memory["SW_params"]["E_p"], label="b_idx")
plt.scatter(sw_class._internal_memory["SW_params"]["f_idx"], sw_class._internal_memory["SW_params"]["E_p"]-input_dict["SW_amplitude"], color="green", label="f_idx")
plt.legend()
#plt.xlim([0,600])
#plt.ylim([-0.01,0])
plt.show()


