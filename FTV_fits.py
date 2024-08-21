import Surface_confined_inference as sci
import numpy as np
import math
results_dict={"FTACV":{"80":{}, "250":{}}, "PSV":{}}
results_dict['FTACV']['80']['3']={'E_start': np.float64(-0.8006475758896721), 'E_reverse': np.float64(-0.0005643140832054527), 'omega': np.float64(2.7893933220163056), 'phase': np.float64(6.0963661768311965), 'delta_E': np.float64(0.07773033393989036), 'v': np.float64(0.0553759383170455)}
results_dict['FTACV']['80']['9']={'E_start': np.float64(-0.8006117990463852), 'E_reverse': np.float64(-0.0005378379312415138), 'omega': np.float64(8.368818213300784), 'phase': np.float64(5.724833637340166), 'delta_E': np.float64(0.06684109310228693), 'v': np.float64(0.05537358012592774)}
results_dict['FTACV']['80']['15']={'E_start': np.float64(-0.8006003712479526), 'E_reverse': np.float64(-0.0005511926856689442), 'omega': np.float64(13.948051829113664), 'phase': np.float64(5.486141199502961), 'delta_E': np.float64(0.054767122705694136), 'v': np.float64(0.055374442627008344)}
results_dict['FTACV']['80']['21']={'E_start': np.float64(-0.8006001654780444), 'E_reverse': np.float64(-0.0005431127034229988), 'omega': np.float64(19.527270389888812), 'phase': np.float64(5.326737618246792), 'delta_E': np.float64(0.044884733179813445), 'v': np.float64(0.055374519015733634),}
results_dict['FTACV']['250']['3']={'E_start': np.float64(-0.8004676428274355), 'E_reverse': np.float64(-0.0005662867260471938), 'omega': np.float64(2.7895996695937977), 'phase': np.float64(6.06881610886128), 'delta_E': np.float64(0.2425898161749251), 'v': np.float64(0.055362889245101125)}
results_dict['FTACV']['250']['9']={'E_start': np.float64(-0.8004057616703748), 'E_reverse': np.float64(-0.0006082039098393999), 'omega': np.float64(8.36881360008673), 'phase': np.float64(5.717100130351908), 'delta_E': np.float64(0.20733790262971458), 'v': np.float64(0.05535907812595028),}
results_dict['FTACV']['250']['15']={'E_start': np.float64(-0.800374111924057), 'E_reverse': np.float64(-0.0006443841911099035), 'omega': np.float64(13.948046784482719), 'phase': np.float64(5.479088647354352), 'delta_E': np.float64(0.16931085920803546), 'v': np.float64(0.055355430054032324)}
results_dict['FTACV']['250']['21']={'E_start': np.float64(-0.8004321886491068), 'E_reverse': np.float64(-0.0006032864694693885), 'omega': np.float64(19.527278389738022), 'phase': np.float64(5.320629887475525), 'delta_E': np.float64(0.1387153181670516), 'v': np.float64(0.055360894910183214),}
results_dict['PSV']['3']={'Edc': np.float64(-0.400924233987332), 'omega': np.float64(2.9977559238552307), 'phase': np.float64(4.480463366514855), 'delta_E': np.float64(0.3867158021828757)}
results_dict['PSV']['9']={'Edc': np.float64(-0.4002382809977385), 'omega': np.float64(8.993290590781472), 'phase': np.float64(4.109126055232471), 'delta_E': np.float64(0.3242408399795645)}
results_dict['PSV']['15']={'Edc': np.float64(-0.40027596880288496), 'omega': np.float64(14.988781419399213), 'phase': np.float64(3.8673457767858714), 'delta_E': np.float64(0.25961556953091686)}
results_dict['PSV']['21']={'Edc': np.float64(-0.4002616829802341), 'omega': np.float64(20.98454610332337), 'phase': np.float64(3.7133684339751927), 'delta_E': np.float64(0.21096337362617562),}
data_dict={"FTACV":{"3":"FTACV_m4D2_PGE_59.60_mV_s-1_3_Hz_250_mV.txt",
                     "9":"FTACV_m4D2_PGE_59.60_mV_s-1_9_Hz_250_mV.txt",
                     "15":"FTACV_m4D2_PGE_59.60_mV_s-1_15_Hz_250_mV.txt",
                     "21":"FTACV_m4D2_PGE_59.60_mV_s-1_21_Hz_250_mV.txt"},
          "PSV":{"3":"PSV_m4D2_PGE_3_Hz_100_osc.txt",
                "9":"PSV_m4D2_PGE_9_Hz_100_osc.txt",
                "15":"PSV_m4D2_PGE_15_Hz_100_osc.txt",
                "21":"PSV_m4D2_PGE_21_Hz_100_osc.txt"}}
loc="/users/hll537/Experimental_data/Interpolated/"
frequencies=["3","9","15","21"]
amps=["80","250"]
for i in range(0,2):
    for j in range(0, len(amps)):
        amp=amps[j]
        for dictionary in [results_dict["PSV"][frequencies[i]], results_dict["FTACV"][amp][frequencies[i]]]:
            dictionary["Temp"]=298
            dictionary["N_elec"]=1
            dictionary["Surface_coverage"]=1e-11
            dictionary["area"]=0.07

        slurm_class = sci.SingleSlurmSetup(
            "FTACV",
            results_dict["FTACV"][amp][frequencies[i]],
            phase_function="constant"
        )
        slurm_class.boundaries = {"k0": [100, 500], 
                            "E0_mean": [-0.45, -0.37],
                            "Cdl": [1e-6, 1e-4],
                            "gamma": [1e-11, 5e-11],
                            "Ru": [1, 500],
                            "E0_std":[0.04, 0.09],
                            "CdlE1":[-3e-3, 3e-3],
                            "CdlE2":[-1e-4, 1e-4],
                            "CdlE3":[-5e-6, 5e-6],
                            "alpha":[0.4, 0.6],
                            "phase":[0,2*math.pi/2],
                            "omega":[0.8*dictionary["omega"], 1.2*dictionary["omega"]],
                            }

        slurm_class.GH_quadrature=True
        slurm_class.dispersion_bins=[25]
        slurm_class.Fourier_fitting=True
        slurm_class.Fourier_window="hanning"
        slurm_class.top_hat_width=0.25
        slurm_class.Fourier_function="abs"
        slurm_class.Fourier_harmonics=list(range(3, 10))
        slurm_class.optim_list = ["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","alpha"]
        file=data_dict["FTACV"][frequencies[i]]
        if amp=="80":
            splitfile=file.split("_")
            splitfile[splitfile.index("250")]="80"
            file="_".join(splitfile)
        slurm_class.setup(
            datafile=loc+"FTACV/{0}/".format(amp)+file,
            cpu_ram="8G",
            time="0-12:00:00",
            runs=20, 
            threshold=1e-8, 
            unchanged_iterations=200,
            #check_experiments={"PSV":{"file":loc+"PSV/"+data_dict["PSV"][frequencies[i]], "parameters":results_dict["PSV"][frequencies[i]]}},
            results_directory=frequencies[i]+"Hz3_FTV_Fourier",
            debug=False,
            run=True
        )
