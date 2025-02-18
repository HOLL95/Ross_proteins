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
                                            square_wave_return="net",
                                            problem="inverse"

                                            )
                times=sw_class.calculate_times()
                potential=sw_class.get_voltage(times)
                netfile=np.savetxt(os.path.join("sw_net_data", file), np.column_stack((list(range(0, len(pot))), data_current, pot)))

                extra_keys=["CdlE1","CdlE2","CdlE3"]
                labels=["constant", "linear","quadratic","cubic"]
                saveloc="/home/userfs/h/hll537/Documents/Experimental_data/SWV_slurm"
                plt.plot(potential)
                plt.scatter(sw_class._internal_memory["SW_params"]["b_idx"], sw_class._internal_memory["SW_params"]["E_p"], s=20, label="b_idx", color="orange")
                plt.scatter(sw_class._internal_memory["SW_params"]["f_idx"], sw_class._internal_memory["SW_params"]["E_p"]+(directions_dict[directions[j]]["v"])*input_dict["SW_amplitude"], s=10, color="red", label="f_idx")
                plt.scatter(sw_class._internal_memory["SW_params"]["b_idx"], pot, s=5, color="green")
                plt.show()
                #newfilelist=file.split(".")
                #newfilelist[-1]="txt"
                #savefile=".".join(newfilelist)
                #np.savetxt(os.path.join(saveloc, savefile),np.column_stack((sw_class._internal_memory["SW_params"]["b_idx"], data_current, pot)),)
                for m in range(0, len(extra_keys)+1):
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
                    sw_class.dispersion_bins=[30]
                    #sw_class.fixed_parameters={"alpha":0.5}
                    sw_class.optim_list=["E0","k0","gamma","Cdl", "alpha"]+extra_keys[:m]
                    sw_class.boundaries={
                        "E0":[-0.46, -0.43],
                        "E0_mean":[-0.46, -0.4],
                        "E0_std":[1e-3, 0.1],
                        "k0":[0.1, 1e3],
                        "alpha":[0.4, 0.6],
                        "gamma":[1e-11, 1e-9],
                        "Cdl":[-10, 10],
                        "alpha":[0.4, 0.6],
                        "CdlE1":[-10, 10]
                        }
                    results=sw_class.Current_optimisation(sw_class._internal_memory["SW_params"]["b_idx"], sw_class.nondim_i(data_current), unchanged_iterations=200, tolerance=1e-8, dimensional=False, parallel=True)
                    print(results)
                    bestfit=sw_class.dim_i(sw_class.simulate(results[:-1],times)) 
                          
                    plt.plot(pot, bestfit)
                    plt.plot(pot, data_current)         
                    plt.show()