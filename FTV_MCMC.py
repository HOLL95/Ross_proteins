import Surface_confined_inference as sci
import numpy as np
import math
import os
experiments_dict={"FTACV":{"3_Hz":{}, "9_Hz":{},"15_Hz":{}, "21_Hz":{}}}
experiments_dict['FTACV']['3_Hz']['80']= {'E_start': np.float64(-0.8902481705989678), 'E_reverse': np.float64(-0.0005663458462876747), 'omega': np.float64(3.0347196895539086), 'phase': np.float64(6.059035018632212), 'delta_E': np.float64(0.07653233662675293), 'v': np.float64(0.05954572063421371)}
experiments_dict['FTACV']['15_Hz']['80']= {'E_start': np.float64(-0.8902605797349086), 'E_reverse': np.float64(-0.0005573884536691498), 'omega': np.float64(15.173689418589497), 'phase': np.float64(5.448960410224393), 'delta_E': np.float64(0.051806426739964634), 'v': np.float64(0.059547949680300125)}
experiments_dict['FTACV']['9_Hz']['80']= {'E_start': np.float64(-0.8902499779033282), 'E_reverse': np.float64(-0.0005430347722881201), 'omega': np.float64(9.10419726283959), 'phase': np.float64(5.692478143186857), 'delta_E': np.float64(0.06450753668694713), 'v': np.float64(0.059548694281080276)}
experiments_dict['FTACV']['21_Hz']['80']= {'E_start': np.float64(-0.8902642585586962), 'E_reverse': np.float64(-0.0005551049836591826), 'omega': np.float64(21.24318369655132), 'phase': np.float64(5.291741324611729), 'delta_E': np.float64(0.04201541965885772), 'v': np.float64(0.05954836661645596)}
experiments_dict['FTACV']['15_Hz']['280']= {'E_start': np.float64(-0.8900404672710482), 'E_reverse': np.float64(-0.0006392975440194792), 'omega': np.float64(15.173686024700986), 'phase': np.float64(5.440366237427825), 'delta_E': np.float64(0.17876387449314424), 'v': np.float64(0.05953022016638514)}
experiments_dict['FTACV']['3_Hz']['280']= {'E_start': np.float64(-0.8901199635261801), 'E_reverse': np.float64(-0.0006379235223668012), 'omega': np.float64(3.03471913390979), 'phase': np.float64(6.05269501468929), 'delta_E': np.float64(0.2672159297976847), 'v': np.float64(0.05953326889446427)}
experiments_dict['FTACV']['21_Hz']['280']= {'E_start': np.float64(-0.8900811295649749), 'E_reverse': np.float64(-0.000625431504669649), 'omega': np.float64(21.24318133865051), 'phase': np.float64(5.285841018363916), 'delta_E': np.float64(0.14497585403067986), 'v': np.float64(0.05953322299348709)}
experiments_dict['FTACV']['21_Hz']['280']= {'E_start': np.float64(-0.8900392326739417), 'E_reverse': np.float64(-0.0006217574272811), 'omega': np.float64(21.42689360258165), 'phase': np.float64(3.770088101350285), 'delta_E': np.float64(0.11428281275681479), 'v': np.float64(0.05953395752776987)}
experiments_dict['FTACV']['9_Hz']['280']= {'E_start': np.float64(-0.8900244598847762), 'E_reverse': np.float64(-0.0006520099910067856), 'omega': np.float64(9.104193706642272), 'phase': np.float64(5.682042161157082), 'delta_E': np.float64(0.22351633655321063), 'v': np.float64(0.059528212300390126)}
results_dict=experiments_dict


loc="/users/hll537/Experimental_data/set2/"
frequencies=[x+"_Hz" for x in ["3","9","15","21"]]
amps=["80","280"]
for i in range(0,len(frequencies)):
    
    for j in range(1, len(amps)):
        amp=amps[j]
        files=os.listdir(loc+"FTACV/"+amp)
        print(type(results_dict["FTACV"][frequencies[i]][amp]))
        for dictionary in [results_dict["FTACV"][frequencies[i]][str(amp)]]:
            dictionary["Temp"]=278
            dictionary["N_elec"]=1
            dictionary["Surface_coverage"]=1e-10
            dictionary["area"]=0.07

        slurm_class = sci.SingleSlurmSetup(
            "FTACV",
            results_dict["FTACV"][frequencies[i]][amp],
            phase_function="constant",            
        )
        slurm_class.boundaries = {"k0": [5, 5000], 
                            "E0_mean": [-0.49, -0.37],
                            "Cdl": [1e-6, 5e-4],
                            "gamma": [1e-11, 8e-10],
                            "Ru": [0.1, 4000],
                            "E0_std":[0.025, 0.15],
                            "CdlE1":[-1e-2, 1e-2],
                            "CdlE2":[-5e-3, 4e-3],
                            "CdlE3":[-5e-5, 5e-5],
                            "alpha":[0.4, 0.6],
                            "phase":[0,2*math.pi/2],
                            "omega":[0.8*dictionary["omega"], 1.2*dictionary["omega"]],
                            }

        slurm_class.GH_quadrature=True
        slurm_class.dispersion_bins=[20]
        slurm_class.Fourier_fitting=True
        slurm_class.Fourier_window="hanning"
        slurm_class.top_hat_width=0.25
        slurm_class.Fourier_function="abs"
        slurm_class.Fourier_harmonics=list(range(4, 10))
        slurm_class.optim_list = ["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","alpha"]
        file=[x for x in files if frequencies[i] in x][0]
        cmaes_loc="/users/hll537/Ross_proteins/inference_results_2_4"
        savefile=frequencies[i]+"_FTV_Fourier_4_"+amp
        dirnames=os.listdir(os.path.join(cmaes_loc, savefile))
        table_loc=os.path.join(cmaes_loc, savefile, [x for x in dirnames if "Pooled" in x][0])
        start=sci._utils.read_param_table(os.path.join(table_loc,"Rounded_table.txt"))[0]
        times=slurm_class.calculate_times(dimensional=False)
        #mcmc_test=slurm_class.dim_i(slurm_class.simulate(start[:-1], times))
        #potential=slurm_class.get_voltage(times, dimensional=True)
        start_dict=dict(zip(slurm_class._optim_list,start[:-1]))
        
        #synthetic_file=np.savetxt("{0}_{1}_synthetic.txt".format(frequencies[i], amp), np.column_stack((slurm_class.dim_t(times), sci._utils.add_noise(mcmc_test, 0.05*max(mcmc_test)), potential)))
        #print(start)
        
        keys=["k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","alpha"]
        fixie={}
        for key in keys:

         fixie[key]=start_dict[key]
        slurm_class.fixed_parameters=fixie
        slurm_class.optim_list=["E0_mean","E0_std"]#"k0", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","alpha"]
       
        new_start=[start_dict[x] for x in slurm_class._optim_list]+[start[-1]]
        print(new_start)
        import matplotlib.pyplot as plt
        data=np.loadtxt(loc+"FTACV/{0}/".format(amp)+file)
        current=data[:,1]
        time=data[:,0]
        
        slurm_class.setup(
            datafile=loc+"FTACV/{0}/".format(amp)+file,#"{0}_{1}_synthetic.txt".format(frequencies[i], amp)
            cpu_ram="12G",
            time="0-48:00:00",
            num_chains=1, 
            samples=30000,
            #transformation={"log":{"k0", "Ru"}},
            starting_point=new_start,
            #CMAES_results_dir=os.path.join(cmaes_loc, savefile, [x for x in dirnames if "Pooled" in x][0]),
            #check_experiments={"PSV":{"file":loc+"PSV/"+data_dict["PSV"][frequencies[i]], "parameters":results_dict["PSV"][frequencies[i]]}},
            results_directory=frequencies[i]+"_FTV_Fourier_data_13_"+amp,
            fixed_sigma=True,
            method="sampling",
            debug=False,
            run=True,
            sigma0=None,
        )
