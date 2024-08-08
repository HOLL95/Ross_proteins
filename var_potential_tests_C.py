import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import math
from scipy.optimize import fmin
psv = sci.SingleExperiment(
    "PSV",
    {
        "Edc":0,
        "omega": 10,
        "delta_E": 0.3,
        "area": 0.07,
        "Temp": 298,
        "N_elec": 1,
        "phase": 0,
        "Surface_coverage": 1e-10,
        #"phase_phase":1,
        #"phase_delta_E":0,
        #"phase_omega":4*np.pi*2
    },
    phase_function="constant"
)
times=psv.calculate_times(PSV_num_peaks=1, dimensional=False)
constant_function={
                    "Edc":0,
                    "omega": 10,
                    "delta_E": 0.3,
                    "area": 0.07,
                    "Temp": 298,
                    "N_elec": 1,
                    "phase": 3*math.pi/2,
                    "Surface_coverage": 1e-10,
                    "phase_phase":0,
                    "phase_delta_E":0,
                    "phase_omega":0,
                    }
sinusoidal_function={
    "Edc":0,
    "omega": 10,
    "delta_E": 0.3,
    "area": 0.07,
    "Temp": 298,
    "N_elec": 1,
    "phase": 0,
    "Surface_coverage": 1e-10,
    "phase_phase":1,
    "phase_delta_E":1,
    "phase_omega":4*np.pi*2
}
psv.fixed_parameters={
                    "E0":0.1,
                    "k0":100,
                    "Ru":100,
                    "gamma":1e-10,
                    "Cdl":1e-4,
                    "alpha":0.5,
                    "cap_phase":0}
fig, ax=plt.subplots()
main_fig, main_ax=plt.subplots()
psv.optim_list=[]
dim_t=psv.dim_t(times)
for amp in [0]:
    sinusoidal_function["phase_delta_E"]=amp
    voltage=psv.get_voltage(dim_t, dimensional=False)
    nondim_e=psv.nondim_e(voltage)
    main_ax.plot(times, nondim_e)
    current=psv.simulate([],times)
    ax.plot(times ,current)


plt.show()
#print([x for x in zip(times[:10], nondim_e[:10])])