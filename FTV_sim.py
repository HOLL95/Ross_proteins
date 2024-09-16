import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
results_dict={"FTACV":{"80":{}, "250":{}}, "PSV":{}}
results_dict['FTACV']['80']['3']={'E_start': np.float64(-0.8006475758896721), 'E_reverse': np.float64(-0.0005643140832054527), 'omega': np.float64(2.7893933220163056), 'phase': np.float64(6.0963661768311965), 'delta_E': np.float64(0.07773033393989036), 'v': np.float64(0.0553759383170455), 'phase_delta_E': np.float64(0.017041489489115325), 'phase_omega': np.float64(0.1), 'phase_phase': np.float64(4.450487852784928)}
results_dict['FTACV']['80']['9']={'E_start': np.float64(-0.8006117990463852), 'E_reverse': np.float64(-0.0005378379312415138), 'omega': np.float64(8.368818213300784), 'phase': np.float64(5.724833637340166), 'delta_E': np.float64(0.06684109310228693), 'v': np.float64(0.05537358012592774), 'phase_delta_E': np.float64(0.0008055595726608189), 'phase_omega': np.float64(395.718786830266), 'phase_phase': np.float64(1.3981535550876787)}
results_dict['FTACV']['80']['15']={'E_start': np.float64(-0.8006003712479526), 'E_reverse': np.float64(-0.0005511926856689442), 'omega': np.float64(13.948051829113664), 'phase': np.float64(5.486141199502961), 'delta_E': np.float64(0.054767122705694136), 'v': np.float64(0.055374442627008344), 'phase_delta_E': np.float64(0.0006014194061334521), 'phase_omega': np.float64(224.77525169930774), 'phase_phase': np.float64(0.8121639027303783)}
results_dict['FTACV']['80']['21']={'E_start': np.float64(-0.8006001654780444), 'E_reverse': np.float64(-0.0005431127034229988), 'omega': np.float64(19.527270389888812), 'phase': np.float64(5.326737618246792), 'delta_E': np.float64(0.044884733179813445), 'v': np.float64(0.055374519015733634), 'phase_delta_E': np.float64(0.0014226254150879747), 'phase_omega': np.float64(78.44189968465425), 'phase_phase': np.float64(1.342048627726686)}
results_dict['FTACV']['250']['3']={'E_start': np.float64(-0.8004676428274355), 'E_reverse': np.float64(-0.0005662867260471938), 'omega': np.float64(2.7895996695937977), 'phase': np.float64(6.06881610886128), 'delta_E': np.float64(0.2425898161749251), 'v': np.float64(0.055362889245101125), 'phase_delta_E': np.float64(0.000896632559170385), 'phase_omega': np.float64(334.6748600982876), 'phase_phase': np.float64(1.5545034546151042)}
results_dict['FTACV']['250']['9']={'E_start': np.float64(-0.8004057616703748), 'E_reverse': np.float64(-0.0006082039098393999), 'omega': np.float64(8.36881360008673), 'phase': np.float64(5.717100130351908), 'delta_E': np.float64(0.20733790262971458), 'v': np.float64(0.05535907812595028), 'phase_delta_E': np.float64(-0.0004392453858779177), 'phase_omega': np.float64(241.2475829845014), 'phase_phase': np.float64(4.017544165276198)}
results_dict['FTACV']['250']['15']={'E_start': np.float64(-0.800374111924057), 'E_reverse': np.float64(-0.0006443841911099035), 'omega': np.float64(13.948046784482719), 'phase': np.float64(5.479088647354352), 'delta_E': np.float64(0.16931085920803546), 'v': np.float64(0.055355430054032324), 'phase_delta_E': np.float64(-0.0003732095390789336), 'phase_omega': np.float64(323.27848934212875), 'phase_phase': np.float64(3.7644641400758823)}
results_dict['FTACV']['250']['21']={'E_start': np.float64(-0.8004321886491068), 'E_reverse': np.float64(-0.0006032864694693885), 'omega': np.float64(19.527278389738022), 'phase': np.float64(5.320629887475525), 'delta_E': np.float64(0.1387153181670516), 'v': np.float64(0.055360894910183214), 'phase_delta_E': np.float64(0.0011065728791195273), 'phase_omega': np.float64(64.53511713089918), 'phase_phase': np.float64(1.3440760913658238)}
results_dict['PSV']['3']={'Edc': np.float64(-0.400924233987332), 'omega': np.float64(2.9977559238552307), 'phase': np.float64(4.480463366514855), 'delta_E': np.float64(0.3867158021828757), 'phase_delta_E': np.float64(-0.005642700425637415), 'phase_omega': np.float64(360.72205406076256), 'phase_phase': np.float64(0.8707021064468491)}
results_dict['PSV']['9']={'Edc': np.float64(-0.4002382809977385), 'omega': np.float64(8.993290590781472), 'phase': np.float64(4.109126055232471), 'delta_E': np.float64(0.3242408399795645), 'phase_delta_E': np.float64(-0.0018387849767278475), 'phase_omega': np.float64(226.04841410221107), 'phase_phase': np.float64(3.186375074140818)}
results_dict['PSV']['15']={'Edc': np.float64(-0.40027596880288496), 'omega': np.float64(14.988781419399213), 'phase': np.float64(3.8673457767858714), 'delta_E': np.float64(0.25961556953091686), 'phase_delta_E': np.float64(0.0011854343898596298), 'phase_omega': np.float64(92.76606968064118), 'phase_phase': np.float64(1.2097269862812539)}
results_dict['PSV']['21']={'Edc': np.float64(-0.4002616829802341), 'omega': np.float64(20.98454610332337), 'phase': np.float64(3.7133684339751927), 'delta_E': np.float64(0.21096337362617562), 'phase_delta_E': np.float64(-0.0003865370447724281), 'phase_omega': np.float64(2.7572035717779357), 'phase_phase': np.float64(0.1714948137654772)}

frequencies=["3","9","15","21"]
amplitudes=["80","250"]
for i in range(0, len(frequencies)):
    
    for j in range(0, len(amplitudes)):
        amp=amplitudes[j]
        dictionary={key:results_dict["FTACV"][amp][frequencies[i]][key] for key in results_dict["FTACV"][amp][frequencies[i]].keys() if "phase_" not in key}
        dictionary["Temp"]=298
        dictionary["area"]=0.07
        dictionary["N_elec"]=1
        dictionary["Surface_coverage"]=1e-11
        slurm_class = sci.SingleSlurmSetup(
            "FTACV",
            dictionary,
            phase_function="constant"
        )
        slurm_class.boundaries = {"k0": [1e-3, 500], 
                            "E0_mean": [-0.6, -0.2],
                            "Cdl": [1e-5, 1e-3],
                            "gamma": [1e-11, 1e-9],
                            "Ru": [1, 1e3],
                            "E0_std":[1e-3, 0.1],
                            "CdlE1":[-3e-3, 3e-3],
                            "CdlE2":[-1e-4, 1e-4],
                            "CdlE3":[-5e-6, 5e-6],
                            "alpha":[0.4, 0.6],
                            "phase":[0,2*math.pi/2],
                            "cap_phase":[0, 2*math.pi/2],
                            "omega":[0.8*results_dict["PSV"][frequencies[i]]["omega"], 1.2*results_dict["PSV"][frequencies[i]]["omega"]],
                            }

        slurm_class.GH_quadrature=True
        slurm_class.dispersion_bins=[30]
        slurm_class.Fourier_fitting=True
        slurm_class.top_hat_width=0.25
        slurm_class.Fourier_function="abs"
        slurm_class.Fourier_harmonics=list(range(3, 10))
        slurm_class.optim_list = ["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","phase", "cap_phase","alpha"]
        init_vals=[-0.4047536814, 0.0507809781,     200.7317584608,   3.5120367467e-11,         351.5388710328,   1.0994474853e-05, 4.1186260389e-04,  7.9306701013e-05,  2.2053550783e-06,  3.0402283637,  3.0546975619,     2.2847792671,     0.5438261746,]
        init_vals=[ -0.4279614654, 0.0588029292,     204.7075623261,   5.9999383440e-11,         721.1722868916,   1.0053023051e-04, -7.9868011396e-03, 4.1358395131e-06,  9.9973984312e-06, dictionary["omega"], dictionary["phase"],  dictionary["phase"],    0.5685817265,]
        #init_vals=[-0.3958839798, 0.0617198064,     236.6690962229,   1.6344329209e-11,         858.1772287593,   3.0016775963e-04, -8.0218142767e-03*0, 4.8175344829e-04*0,  3.9342917225e-06*0, dictionary["omega"], dictionary["phase"],  dictionary["phase"],    0.4739785866,]
        init_vals=[-0.4352443352, 0.0609219863,     400.6347259279,   4.4072157623e-11,         59.950040209,     6.1135543667e-05, -8.5551500437e-03, 9.9634405758e-04,  2.6833205674e-05,  dictionary["omega"], dictionary["phase"],  dictionary["phase"],      0.5310420538,]
        #slurm_class.optim_list = ["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","alpha"]
        
        init_vals=[ -0.4300707971, 0.0400286827,     335.0931379793,   4.9995656981e-11,         50.9878516972,    9.6151652803e-05, -4.2647495067e-05, 4.9986311987e-05,  2.1174990364e-07,  2.7895549111, 0,0, 0.4599594539,]
        init_vals=[-0.4358444576, 0.0400020508,     499.2308820791,   4.9999992590e-11,         1.0197771743,     9.9994944115e-05, 2.3072684826e-03,  9.9999056053e-05,  -4.9756151692e-06, 2.78974092, 0,0,   0.5228125818,]
        data=np.loadtxt("/home/henryll/Documents/Experimental_data/ForGDrive/Interpolated/FTACV/{1}/FTACV_m4D2_PGE_59.60_mV_s-1_{0}_Hz_{1}_mV.txt".format(frequencies[i], amp))
        time=data[:,0]
        potential=data[:,2]
        current=data[:,1]
        test=slurm_class.dim_i(slurm_class.Dimensionalsimulate(init_vals, time))
        plt.plot(slurm_class.experiment_top_hat(time, current))
        plt.plot(slurm_class.experiment_top_hat(time, test))
        plt.show()
        axes=sci.plot.plot_harmonics(Data_data={"time":time, "current":current*1e6, "potential":potential, "harmonics":list(range(2, 10))},
                                Sim_data={"time":time, "current":test*1e6, "potential":potential, "harmonics":list(range(2, 10))},plot_func=np.abs, 
                                hanning=True, filter_val=0.5, xlabel="Time (s)", ylabel="Current ($\\mu$A)", remove_xaxis=True)
        axes[0].set_title("{1} mV {0} Hz".format(frequencies[i], amp))
        plt.show()
       
    #fig=plt.gcf()
    #fig.set_size_inches(5, 6)
    #fig.savefig("Initial_plots/Init_results/{0}_Hz_FTV_harmonics".format(frequencies[i]), dpi=500)
