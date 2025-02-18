import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
loc="/home/henryll/Documents/Experimental_data/Nat/m4d2_SET3"
files =os.listdir(loc)
results={"anodic":{},"cathodic":{}}
freqs=[25, 65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
from numpy.lib.stride_tricks import sliding_window_view

strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
directions_map=dict(zip(directions, ["1.csv", "2.csv"]))
fig, ax=plt.subplots(2,2)
paramloc="/home/henryll/Documents/Inference_results/swv/set6_1"
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}
smooths=[0, 5, 10, 15, 20]
scatter_points=[]
for o in range(0, len(smooths)):
    smoothing=smooths[o]
    for i in range(1, len(freqs)):
        for j in range(0, len(directions)):
            for file in files:
                if "_{0}_".format(strfreqs[i]) in file and directions_map[directions[j]] in file:
                    splitfile=file.split("_")
                    print(splitfile)
                    if "lock" not in file:
                        hzdx=splitfile.index("Hz")
                        
                        frequency=freqs[i]
                    
                        try:
                            data=np.loadtxt(os.path.join(loc,file), delimiter=",")
                        except:
                            data=np.loadtxt(os.path.join(loc,file))
                        pot=data[:-1,0]
                        data_current=data[:-1,1]
                        best_loc=np.where((pot>-0.6) & (pot<-0.3))
                        best_fit=sci._utils.read_param_table(os.path.join(paramloc, "SWV_{0}".format(freqs[i]), directions[j], "cubic".lower(), "PooledResults_2024-11-18","Full_table.txt"))[0]
                    
                        sw_class=sci.SingleSlurmSetup("SquareWave",
                                    {"omega":freqs[i],
                                    "scan_increment":5e-3,#abs(pot[1]-pot[0]),
                                    "delta_E":0.8,
                                    "SW_amplitude":5e-3,
                                    "sampling_factor":200,
                                    "E_start":directions_dict[directions[j]]["E_start"],
                                    "Temp":278,
                                    "v":directions_dict[directions[j]]["v"],
                                    "area":0.036,
                                    "N_elec":1,
                                    "Surface_coverage":1e-10},
                                    square_wave_return="net",
                                    problem="forwards"

                                    )
                        sw_class.dispersion_bins=[1]
                        sw_class.optim_list=["E0_mean", "E0_std","k0","gamma","Cdl", "alpha"]+["CdlE1","CdlE2","CdlE3"]
                        best_fit[2]=0
                        line=sw_class.dim_i(sw_class.simulate(best_fit[:-1], sw_class.calculate_times()))
                        #plt.plot(data_current-line)
                        if freqs[i]==100:
                            if o==0:
                                if j==0:
                                    ax[0,0].plot(pot, data_current, color=sci._utils.colours[0], label="Experimental data")
                                    ax[0,0].plot(pot, line, color="black", linestyle="--", label="Estimated cubic background")
                                else:
                                    ax[0,0].plot(pot, data_current, color=sci._utils.colours[0], )
                                    ax[0,0].plot(pot, line, color="black", linestyle="--",)

                        #plt.show()
                        line=np.mean(data_current[np.where(pot<-0.7)])
                        data_current=np.subtract(data_current, line)
                        if smoothing!=0:
                            data_current=sci._utils.moving_avg(data_current, smoothing)
                        #plt.plot(pot, data_current)
                        #plt.show()
                        if directions_map["cathodic"] in file:
                            key="cathodic"
                            results["cathodic"][frequency]=min(data_current[best_loc])
                        elif directions_map["anodic"] in file:
                            key="anodic"
                            results["anodic"][frequency]=max(data_current[best_loc])
                    

                        datadx=np.where(data_current== results[key][frequency])
                        if freqs[i]==100:
                            if j==0:
                                ax[0,1].plot(pot, data_current/freqs[i], label=smoothing, color=sci._utils.colours[o])
                            else:
                                 ax[0,1].plot(pot, data_current/freqs[i], color=sci._utils.colours[o])
                            scatter_points.append([pot[datadx][0], results[key][frequency]/freqs[i]])
                           
                            ax[0,1].set_xlabel("Potential (V)")
                            ax[0,1].set_ylabel("Scaled current (A Hz$^{-1}$)")
                            
                            
    ax[0,1].legend()
                

    frequencies=sorted(results["anodic"].keys())
    for key in ["anodic", "cathodic"]:
        for normed in [True, False]:
            if normed==True:
                
                values=[results[key][x]/x for x in frequencies ]
                axis=ax[1,0]
            else:
                values=[results[key][x] for x in frequencies ]
                axis=ax[1,1]

            
            if key=="anodic":
            
                axis.plot(frequencies, values, color=sci._utils.colours[o], linestyle="--",)
            else:
                axis.plot(frequencies, values, color=sci._utils.colours[o])
#for i in range(0, len(scatter_points)):
#    ax[0,0].scatter(scatter_points[i][0], scatter_points[i][1], edgecolors="black", color=sci._utils.colours[i//2], s=50)
ax[1,0].plot(25, 0, color="black", linestyle="--", label="Anodic")
ax[1,0].plot(25, 0, color="black",  label="Cathodic")
ax[1,0].legend()
ax[0,1].set_title("100 Hz background subtracted")
ax[1,0].set_xlabel("Frequency (Hz)")
ax[1,0].set_ylabel("Scaled current (A Hz$^{-1}$)")
ax[1,0].set_title("Scaled currents of greatest magnitude ")
ax[1,1].set_xlabel("Frequency (Hz)")
ax[1,1].set_ylabel("Max current (A)")
ax[1,1].set_title("Currents of greatest magnitude")
ax[0,0].legend()
ax[0,0].set_title("100 Hz")
ax[0,0].set_xlabel("Potential (V)")
ax[0,0].set_ylabel("Current (A)")
plt.legend()
plt.show()
fig.savefig("peakmax.png", dpi=500)