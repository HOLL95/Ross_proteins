import numpy as np
import matplotlib.pyplot as plt
import Surface_confined_inference as sci
import os
loc="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Trumpet/"

"""sci.HeuristicMethod(os.listdir(loc),
                    method="DCVTrumpet",
                    area=0.07,
                    data_loc=loc,
                    save_address="M4D2_positions2.txt",
                    fitting=False,
                    header_length=1)"""

data=np.loadtxt(loc, skiprows=1)
time=data[:,0]
current=data[:,2]
params={"E_start":-0.8,"E_reverse":0,"v":150e-3,"area":0.07, "Temp":278,"N_elec":1,"Surface_coverage":1e-10}
sim=sci.SingleExperiment("DCV",params)
times=sim.calculate_times()




sim.fixed_parameters={"gamma":1e-10, "Cdl":1e-4, "Ru":20000}
sim.optim_list=["E0", "k0","alpha"]

sim_vals=sim.simulate([-0.4, 10,0.5],times)
plt.plot(sim.dim_t(times), sim.dim_i(sim_vals))
plt.plot(time, current)
plt.show()



