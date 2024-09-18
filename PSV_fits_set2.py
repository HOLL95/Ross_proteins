import Surface_confined_inference as sci
import numpy as np
import math
import os
experiments_dict={"FTACV":{"3_Hz":{}, "9_Hz":{},"15_Hz":{}, "21_Hz":{}}}
experiments_dict['PSV']['21_Hz']['100_osc']= {'Edc': -0.4451549843492726, 'omega': 21.24332547828609, 'phase': 3.7061559843240985, 'delta_E': 0.23032103891330358}
experiments_dict['PSV']['3_Hz']['100_osc']= {'Edc': -0.44504922351594134, 'omega': 3.0347403892751337, 'phase': 4.4766643586691055, 'delta_E': 0.42921518128386493}
experiments_dict['PSV']['9_Hz']['100_osc']= {'Edc': -0.4450602562455672, 'omega': 9.104241058699305, 'phase': 4.1018217711950955, 'delta_E': 0.3575878846032547}
experiments_dict['PSV']['15_Hz']['100_osc']= {'Edc': -0.44510516159029523, 'omega': 15.173801327488201, 'phase': 3.8599820604984774, 'delta_E': 0.28491327761894464}
results_dict=experiments_dict


loc="/users/hll537/Experimental_data/set2/"
frequencies=[x+"_Hz" for x in ["3","9","15","21"]]

for i in range(0,len(frequencies)):
    
   
   
    files=os.listdir(loc+"PSV/")
    print(type(results_dict["PSV"][frequencies[i]]["100_osc"]))
    for dictionary in [results_dict["FTACV"][frequencies[i]][str(amp)]]:
        dictionary["Temp"]=278
        dictionary["N_elec"]=1
        dictionary["Surface_coverage"]=1e-10
        dictionary["area"]=0.07

    slurm_class = sci.SingleSlurmSetup(
        "PSV",
        results_dict["PSV"][frequencies[i]]["100_osc"],
        phase_function="constant"
    )
    slurm_class.boundaries = {"k0": [5, 5000], 
                        "E0_mean": [-0.45, -0.37],
                        "Cdl": [1e-6, 5e-4],
                        "gamma": [1e-11, 8e-10],
                        "Ru": [0.1, 4000],
                        "E0_std":[0.025, 0.15],
                        "CdlE1":[-1e-2, 1e-2],
                        "CdlE2":[-5e-3, 4e-3],
                        "CdlE3":[-5e-5, 5e-5],
                        "alpha":[0.4, 0.6],
                        "phase":[0,2*math.pi/2],
                        "cap_phase":[0,2*math.pi/2],
                        "omega":[0.8*dictionary["omega"], 1.2*dictionary["omega"]],
                        }

    slurm_class.GH_quadrature=True
    slurm_class.dispersion_bins=[30]
    slurm_class.Fourier_fitting=True
    slurm_class.Fourier_window="hanning"
    slurm_class.top_hat_width=0.25
    slurm_class.Fourier_function="inverse"
    slurm_class.Fourier_harmonics=list(range(4, 10))
    slurm_class.optim_list = ["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","alpha", "cap_phase"]
    file=[x for x in files if frequencies[i] in x][0]
    
    slurm_class.setup(
        datafile=loc+"PSV/"+file,
        cpu_ram="12G",
        time="0-12:00:00",
        runs=20, 
        threshold=1e-8, 
        unchanged_iterations=200,
        #check_experiments={"PSV":{"file":loc+"PSV/"+data_dict["PSV"][frequencies[i]], "parameters":results_dict["PSV"][frequencies[i]]}},
        results_directory=frequencies[i]+"_PSV_Fourier_1_"+amp,
        save_csv=False,
        debug=False,
        run=True
    )
