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
loc="/home/henryll/Documents/Experimental_data/ForGDrive/Interpolated/"
frequencies=["3","9","15","21"]
for i in range(0,len(frequencies)):
    for dictionary in [results_dict["PSV"][frequencies[i]], results_dict["FTACV"]["250"][frequencies[i]]]:
     dictionary["Temp"]=298
     dictionary["N_elec"]=1
     dictionary["Surface_coverage"]=1e-11
     dictionary["area"]=0.07

    slurm_class = sci.SingleSlurmSetup(
        "PSV",
        results_dict["PSV"][frequencies[i]],
        phase_function="constant"
    )
    slurm_class.boundaries = {"k0": [1e-3, 500], 
                        "E0_mean": [-0.45, -0.35],
                        "Cdl": [1e-6, 1e-4],
                        "gamma": [1e-11, 6e-11],
                        "Ru": [1, 1e3],
                        "E0_std":[1e-3, 0.06],
                        "CdlE1":[-8e-3, 8e-3],
                        "CdlE2":[-5e-4, 5e-4],
                        "CdlE3":[-1e-5, 1e-5],
                        "alpha":[0.4, 0.6],
                        "phase":[0,2*math.pi/2],
                        "cap_phase":[0, 2*math.pi/2],
                        "omega":[0.8*results_dict["PSV"][frequencies[i]]["omega"], 1.2*results_dict["PSV"][frequencies[i]]["omega"]],
                        }

    slurm_class.GH_quadrature=True
    slurm_class.dispersion_bins=[16]
    slurm_class.Fourier_fitting=True
    slurm_class.Fourier_window="hanning"
    slurm_class.top_hat_width=0.5
    slurm_class.Fourier_function="composite"
    slurm_class.Fourier_harmonics=list(range(3, 10))
    slurm_class.optim_list = ["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","phase", "cap_phase","alpha"]
    vals=[ -0.4473366371, 1.3838214001e-02, 1.613257081,      3.6422422587e-11,         641.1116274629,   7.5122274406e-05, 9.8522734385e-04,  3.0663524614e-04,  9.4027716834e-06,  15.0411609198, 0.0749376784,     2.3728325549,    0.4952567078, ]
    vals=[ -0.3960441066, 6.5760581047e-03, 129.030442357,    2.0943823265e-11,         799.9715982521,   8.3457881706e-04*0.15, 7.6698755033e-03,  3.5920518897e-04,  -3.4905220018e-06, dictionary["omega"],  dictionary["phase"],    dictionary["phase"],     0.5567709679]
    vals=[-0.3939129035, 0.0299096422,     100.7564518212,   2.9160858569e-11,         626.8050251157,   7.3227086473e-05, 1.0915075973e-03,  -4.7015409912e-04, 3.9257691660e-06, 4.7257261124, 2.9838114223,  4.859606171,     0.5587321521]
    vals=[-0.4279614654, 0.0588029292,     204.7075623261,   5.9999383440e-11,         721.1722868916,   1.0053023051e-04, -7.9868011396e-03, 4.1358395131e-06,  9.9973984312e-06, 4.3557540068, 2.9977399043,  4.2546950225,    0.5685817265, ]
    data=np.loadtxt(loc+"PSV/"+data_dict["PSV"][frequencies[i]])
    import matplotlib.pyplot as plt
    
    time=data[:,0]
    time_idx=np.where(time>(5/(results_dict["PSV"][frequencies[i]]["omega"])))
    current=data[:,1]
    potential=data[:,2]
    sim=slurm_class.dim_i(slurm_class.Dimensionalsimulate(vals, time))
    sci.plot.plot_harmonics(data_data={"time":time[time_idx], "current":current[time_idx], "potential":potential[time_idx], "harmonics":list(range(3, 10))},
                            sim_data={"time":time[time_idx], "current":sim[time_idx], "potential":potential[time_idx], "harmonics":list(range(3, 10))}, xaxis="potential"
    )
    fig, ax=plt.subplots()
    ax.plot(potential, current)
    ax.plot(potential, sim)
    plt.show()
