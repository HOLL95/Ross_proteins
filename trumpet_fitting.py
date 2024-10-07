import numpy as np
import matplotlib.pyplot as plt
import Surface_confined_inference as sci
import os
loc="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Trumpet"
"""
sci.HeuristicMethod(os.listdir(loc),
                    method="DCVTrumpet",
                    area=0.07,
                    data_loc=loc,
                    save_address="M4D2_positions.txt",
                    fitting=False,
                    header_length=1)
"""
positions=np.loadtxt("M4D2_positions.txt",delimiter=",")
scan_rates=positions[:,0]/1e3

anodic=positions[:,1]
cathodic=positions[:,2]
data=np.column_stack((anodic, cathodic))
params={"E_start":-0.8,"E_reverse":0,"v":150e-3,"area":0.07, "Temp":278,"N_elec":1,"Surface_coverage":1e-10}


sim=sci.TrumpetSimulator(params, scan_rates=scan_rates,sampling_factor=2000)
sim.fixed_parameters={"gamma":1e-10, "Cdl":1e-4, "Ru":1}
sim.optim_list=["E0", "k0","alpha", "dcv_sep"]
fig,axes=plt.subplots(1,3)

counter=0
for cdl in [5e-5, 7.5e-5, 1e-4]:
        ax=axes[counter]
        ax.set_xlabel("Log (v)")
        ax.set_ylabel("Peak position (V)")
        ax.set_title("CDL={0}".format(cdl))
        sim.trumpet_plot(scan_rates, data,ax=ax)
        counter+=1
        for Ru in [200, 1000, 2000, 10000]:
        
                sim.fixed_parameters={"gamma":8e-11, "Cdl":cdl, "Ru":Ru}
                sim_vals=sim.TrumpetSimulate([np.float64(-0.43859711622186015), np.float64(15.719941331822666), np.float64(0.5999580453661827), np.float64(0.00047060618658137837)],scan_rates)
                sim.trumpet_plot(scan_rates, sim_vals,ax=ax, label="R={0}".format(Ru))
        ax.legend()      
plt.show()
sim.fit(scan_rates, data, 
        plot_results=True, 
        unchanged_iterations=200,                 
        threshold=1e-4, 
        boundaries={"E0":[-0.48, -0.4], "k0":[0.1, 500],"alpha":[0.4, 0.6],"dcv_sep":[0, 0.1]},
        parallel=True)
ax=plt.gca()




