import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import os
loc="/home/henryll/Documents/Experimental_data/Nat/m4d2_SET3"
files =os.listdir(loc)

freqs=[25, 65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300,  400, 500]
strfreqs=[str(x) for x in freqs]
directions=["anodic","cathodic"]
directions_map=dict(zip(directions, ["1.csv", "2.csv"]))
paramloc="/home/henryll/Documents/Inference_results/swv/set4_10"
directions_dict={"anodic":{"v":1, "E_start":-0.8},"cathodic":{"v":-1, "E_start":0}}

for i in range(1, len(freqs)):
    for j in range(0, len(directions)):
        for file in files:
            if "_{0}_".format(strfreqs[i]) in file and directions_map[directions[j]] in file:
                try:
                    data=np.loadtxt(os.path.join(loc, file), delimiter=",")
                except:
                    data=np.loadtxt(os.path.join(loc, file))
                pot=data[:-1, 0]
                data_current=data[:-1,1]
                #plt.plot(pot, data_current)
                #plt.show()
                input_dict= {"omega":freqs[i],
                                            "scan_increment":5e-3,#abs(pot[1]-pot[0]),
                                            "delta_E":0.8,
                                            "SW_amplitude":5e-3,
                                            "sampling_factor":200,
                                            "E_start":directions_dict[directions[j]]["E_start"],
                                            "Temp":278,
                                            "v":directions_dict[directions[j]]["v"],
                                            "area":0.036,
                                            "N_elec":1,
                                            "Surface_coverage":1e-10}
                sw_class=sci.SingleSlurmSetup("SquareWave",
                                            input_dict,
                                            square_wave_return="total",
                                            problem="forwards"

                                            )
                times=sw_class.calculate_times()
                potential=sw_class.get_voltage(times)
                netfile=np.savetxt(os.path.join("sw_net_data", file), np.column_stack((list(range(0, len(pot))), data_current, pot)))

                extra_keys=["CdlE1","CdlE2","CdlE3"]
                labels=["constant", "linear","quadratic","cubic"]
                saveloc="/home/userfs/h/hll537/Documents/Experimental_data/SWV_slurm"
               
                m=0
                sw_class.optim_list=["E0","k0","gamma","Cdl", "alpha"]+extra_keys[:m]
                fig, ax=plt.subplots(1, 2)
                ax[0].plot(potential, label="Total")
                ax[0].scatter(sw_class._internal_memory["SW_params"]["b_idx"], sw_class._internal_memory["SW_params"]["E_p"], s=20, label="Backwards", color="orange")
                ax[0].scatter(sw_class._internal_memory["SW_params"]["f_idx"], sw_class._internal_memory["SW_params"]["E_p"]+(directions_dict[directions[j]]["v"])*input_dict["SW_amplitude"], s=10, color="red", label="Forwards")
                
                values=[np.float64(-0.43000000000000027), np.float64(50.748999187158979), np.float64(9.999999999999862e-10), np.float64(0.9092870723248012), np.float64(0.5063600884630746), np.float64(0.05120377189812185)]
                total=-1*sw_class.dim_i(sw_class.Dimensionalsimulate(values[:-1], times))
                _,_,net,_=sw_class.SW_peak_extractor(total)
                ax[1].plot(total, label="Total")
                points=[sw_class._internal_memory["SW_params"]["b_idx"],sw_class._internal_memory["SW_params"]["f_idx"]]
                cs=["orange","red"]
                labels=["Backwards","Forwards"]
                sets=[]
                for h in range(0,2):
                    yplt= [total[int(x)] for x in points[h]]
                    ax[1].scatter(points[h],yplt, s=20, color=cs[h], label=labels[h])
                    ax[h].set_xlim([14000, 20000])
                    ax[h].set_xlabel("Dimensionless time")
                ax[0].set_ylim([-0.45, -0.3])
                net_positions=[x+100 for x in points[0] ]
                print(net_positions)
                ax[1].plot(net_positions, -1*net, linestyle="--", color="black", label="Net")
                ax[0].legend()
                ax[1].legend()
                ax[0].set_ylabel("Potential (V)")
                ax[1].set_ylabel("Current (A)")
               
               
                plt.show()
                fig.savefig("sampling_example.png", dpi=500)

                       
                  