import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
loc="/home/henryll/Documents/Experimental_data/Nat/m4d2_SET3"
files =os.listdir(loc)
results={"anodic":{},"cathodic":{}}
freqs=[25, 65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
directions_map=dict(zip(directions, ["1.csv", "2.csv"]))
fig, ax=plt.subplots(1,3)
paramloc="/home/henryll/Documents/Inference_results/swv/set4_10"

for i in range(0, len(freqs)):
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
                    #best_fit=sci._utils.read_param_table(os.path.join(paramloc, "SWV_{0}".format(freqs[i]), directions[j], "linear".lower(), "PooledResults_2024-11-08","Full_table.txt"))[0]
                    line=np.mean(data_current[np.where(data_current>-0.7)])
                    """sw_class=sci.SingleSlurmSetup("SquareWave",
                                {"omega":freqs[i],
                                "scan_increment":2e-3,#abs(pot[1]-pot[0]),
                                "delta_E":0.8,
                                "SW_amplitude":2e-3,
                                "sampling_factor":200,
                                "E_start":-0.8,
                                "Temp":278,
                                "v":-1,
                                "area":0.036,
                                "N_elec":1,
                                "Surface_coverage":1e-10},
                                square_wave_return="net",
                                problem="forwards"

                                )
                    """
                    data_current=np.subtract(data_current, line)
                    if directions_map["cathodic"] in file:
                        key="cathodic"
                        results["cathodic"][frequency]=min(data_current[best_loc])
                    elif directions_map["anodic"] in file:
                        key="anodic"
                        results["anodic"][frequency]=max(data_current[best_loc])
                   

                    datadx=np.where(data_current== results[key][frequency])
                    ax[j].plot(pot, data_current/freqs[i], label=freqs[i])
                    ax[j].scatter(pot[datadx][0], results[key][frequency]/freqs[i], edgecolors="black")
                    ax[j].set_xlabel("Potential")
                    ax[j].set_ylabel("Current/frequency")
                    ax[j].set_title(directions[j])
ax[0].legend()
              

frequencies=sorted(results["anodic"].keys())
for key in ["anodic", "cathodic"]:
    values=[results[key][x]/x for x in frequencies ]
    ax[2].plot(frequencies, values, label=key)

plt.legend()
plt.show()