import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import math
from scipy.optimize import fmin
import copy
master_param={
    'E_start': np.float64(-0.8004676428274355), 
    'E_reverse': np.float64(-0.0005662867260471938), 
    'omega': np.float64(2.7895996695937977), 
    'phase': np.float64(6.06881610886128), 
    'delta_E': np.float64(0.2425898161749251), 
    'v': np.float64(0.055362889245101125), 
    'phase_delta_E': np.float64(0.000896632559170385), 
    'phase_omega': np.float64(334.6748600982876), 
    'phase_phase': np.float64(1.5545034546151042),
    'Temp':298,
    'N_elec':1,
    'area':0.07,
    'Surface_coverage':1e-10,
    }



multiple=[0.5, 1, 1.5]
params=["phase_"+x for x in ["delta_E","omega","phase"]]

fig,ax=plt.subplots(1,3)
for i in range(0, len(params)):
    for j in range(0, len(multiple)):
        input_parameters=copy.deepcopy(master_param)

        input_parameters[params[i]]=master_param[params[i]]*multiple[j]
        ftv = sci.SingleExperiment(
        "FTACV",
        input_parameters,
        phase_function="sinusoidal"
        )
        times=ftv.calculate_times( dimensional=False)

        ftv.fixed_parameters={
                            "E0":-0.5,
                            "k0":100,
                            "Ru":100,
                            "gamma":1e-10,
                            "Cdl":1e-4,
                            "alpha":0.5,
                            }

      
        ftv.optim_list=[]

        current=ftv.simulate([],times)
        ax[i].plot(times ,current, label=multiple[i])


plt.show()
#print([x for x in zip(times[:10], nondim_e[:10])])