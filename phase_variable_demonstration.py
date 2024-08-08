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
   
    'Temp':298,
    'N_elec':1,
    'area':0.07,
    'Surface_coverage':1e-11,
    }



multiple=[0.5, 1, 1.5]
params=["phase_"+x for x in ["delta_E","omega","phase"]]


sinusoidal_params=copy.deepcopy(master_param)
sinusoidal_params['phase_delta_E']= np.float64(0.000896632559170385)
sinusoidal_params['phase_omega']=  np.float64(334.6748600982876)
sinusoidal_params['phase_phase']= np.float64(1.5545034546151042)
values=["constant","sinusoidal"]
inputs=[master_param, sinusoidal_params]
currents=[]
for i in range(0,len(values)):

    ftv = sci.SingleExperiment(
    "FTACV",
    inputs[i],
    phase_function=values[i]
    )
    times=ftv.calculate_times(sampling_factor=400, dimensional=False)

    ftv.fixed_parameters={
                        "E0":-0.5,
                        "k0":100,
                        "Ru":100,
                        "gamma":5e-12,
                        "Cdl":1e-4,
                        "alpha":0.5,
                        "CdlE1":3e-3,
                        "CdlE2":-5e-4,
                        "CdlE3":1e-6
                        }


    ftv.optim_list=[]

    current=ftv.simulate([],times)
    plt.plot(times ,current, label=multiple[i])
    currents.append({"time":times, "current":current, "harmonics":list(range(1, 10))})
sci.plot.plot_harmonics(Constant_phase_data=currents[0], Sinusoidal_phase_data=currents[1],hanning=True, plot_func=np.abs, filter_val=0.2)# 
plt.show()
#print([x for x in zip(times[:10], nondim_e[:10])])