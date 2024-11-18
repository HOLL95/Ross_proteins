import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
loc="/home/userfs/h/hll537/Documents/Experimental_data/SWV"
loc="/users/hll537/Experimental_data/SWV"
freqs=np.arange(25, 1100, 10)
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
#files=os.listdir(loc)
k0_vals=np.array([100, 200, 300, 500, 750, 1250])
fig, ax=plt.subplots(2, 2)
freqmax=np.zeros((2, len(k0_vals)))
for g in range(0, len(k0_vals)):
    peakmax=np.zeros((2, len(freqs)))

    for i in range(0,len(freqs)):
       
        for j in range(0, len(directions)):
            #file=[x for x in files if (strfreqs[i] in x and directions[j] in x and ".csv" in x)][0]
            """try:
                data=np.loadtxt(os.path.join(loc, file), delimiter=",")
            except:
                data=np.loadtxt(os.path.join(loc, file))"""
            #pot=data[:-1, 0]
            #data_current=data[:-1,2]
            #plt.plot(pot, data_current)
            #plt.show()
            sw_class=sci.SingleSlurmSetup("SquareWave",
                                        {"omega":freqs[i],
                                        "scan_increment":10e-3,#abs(pot[1]-pot[0]),
                                        "delta_E":0.8,
                                        "SW_amplitude":10e-3,
                                        "sampling_factor":200,
                                        "E_start":directions_dict[directions[j]]["E_start"],
                                        "Temp":278,
                                        "v":directions_dict[directions[j]]["v"],
                                        "area":0.036,
                                        "N_elec":1,
                                        "Surface_coverage":1e-10},
                                        square_wave_return="net",
                                        problem="inverse"

                                        )
            times=sw_class.calculate_times()
            potential=sw_class.get_voltage(times)
            #netfile=np.savetxt(os.path.join("sw_net_data", file), np.column_stack((list(range(0, len(pot))), data_current, pot)))

            extra_keys=["CdlE1"]
            labels=["constant", "linear","quadratic","cubic"]
            saveloc="/home/userfs/h/hll537/Documents/Experimental_data/SWV_slurm"
            #newfilelist=file.split(".")
            #newfilelist[-1]="txt"
            #savefile=".".join(newfilelist)
            #np.savetxt(os.path.join(saveloc, savefile),np.column_stack((sw_class._internal_memory["SW_params"]["b_idx"], data_current, pot)),)
           
            sw_class.boundaries={
            "E0":[-0.5, -0.3],
            "E0_mean":[-0.5, -0.3],
            "E0_std":[1e-3, 0.1],
            "k0":[0.1, 5e3],
            "alpha":[0.4, 0.6],
            "gamma":[1e-11, 1e-8],
            "Cdl":[-10,10],
            "CdlE1":[-10, 10],
            "CdlE2":[-10, 10],
            "CdlE3":[-10, 10],
            "alpha":[0.4, 0.6]
            }
            sw_class.fixed_parameters={"alpha":0.4}
            sw_class.optim_list=["E0","k0","gamma","Cdl"]+extra_keys

            values=[-0.4, k0_vals[g], 1e-10, 0.5, 0]
            sim=sw_class.Dimensionalsimulate(values, times)
            if j==1:
                func=min
            else:
                func=max
            peakmax[j, i]=func(sim)
           
            
    for v in range(0, 2):
        ax[v, 0].plot(freqs, peakmax[v,:], label="k={0}".format(k0_vals[g]))

        ax[v, 0].set_ylabel("Normalised current")  
        ax[v, 0].set_xlabel("Frequency (Hz)")
        if v==1:
            func=min
        else:
            func=max
        abspeak=abs(peakmax[v,:])
        maxfreq=freqs[np.where(abspeak==func(abspeak))][0]
        freqmax[v, g]=maxfreq
        ax[v, 0].scatter(maxfreq, func(abspeak))
from scipy import stats
for v in range(0, 2):
    ax[v, 1].scatter(k0_vals, freqmax[v, :])

    slope, intercept, r_value, p_value, std_err = stats.linregress(k0_vals,freqmax[v,:])
    line=slope*k0_vals+intercept
    ax[v, 1].plot(k0_vals, line)
    ax[v, 1].set_xlabel("k0 value (s$^{-1}$)")
    ax[v, 1].set_ylabel("Frequency where max I observed (Hz)")
    print(intercept, slope, (175-intercept)/slope)
    print(intercept, slope, (200-intercept)/slope)
    print(intercept, slope, (225-intercept)/slope)
ax[0,0].legend()
plt.show()