import Surface_confined_inference as sci
import numpy as np
import math
import os
import matplotlib.pyplot as plt
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
loc="/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/Interpolated/FTACV"

frequencies=[x+"_Hz" for x in ["3","9","15"]]
best_fits=dict(zip(frequencies,[
    [-0.4244818992, 0.0417143285,     4.9920932937e+03, 1.8134851259e-10,         2.5644285241e+03, 6.6014432067e-05, -4.6443012010e-03, -1.2340727286e-03, -1.4626502188e-05, 3.0346809949,  0.524292126,],
    [-0.4205770335, 0.0650104188,     1.8457772599e+03, 1.1283378372e-10,         764.9843288034,   9.9054277034e-05, -3.5929255146e-03, -7.1997893984e-04, -5.3937670998e-06, 9.1041692039,  0.4770065713,     ],
    [-0.4225575542, 0.0765111705,     111.5784183264,   7.7951293716e-11,         184.9121615528,   9.9349322658e-05, -1.6383464471e-03, -3.3463983363e-04, -7.5051594548e-06, 15.1735928635, 0.4006210586, ]

]))
nondims={
    "3_Hz":{'E_start': np.float64(-37.15617543627824), 'E_reverse': np.float64(-0.026628768349483585), 'omega': np.float64(7.672744209895916), 'phase': 0, 'delta_E': np.float64(11.15436387652591), 'v': np.float64(1.0), 'Temp': 278, 'N_elec': 1, 'Surface_coverage': 1e-10, 'area': 0.07, 'CdlE2': -0.0012340727286, 'gamma': 1.8134851259, 'CdlE3': -1.4626502188e-05, 'Cdl': np.float64(0.16390610439952177), 'alpha': 0.524292126, 'E0_std': 1.741276425047796, 'CdlE1': -0.004644301201, 'k0': np.float64(2008.8175776654834), 'E0_mean': -17.71909918042849, 'Ru': np.float64(0.17966959749510292), 'E0': 0.0, 'theta': 0, 'phase_flag': 0, 'Marcus_flag': 0, 'tr': np.float64(37.12954666792876), 'cap_phase': 0},
    "9_Hz":{'E_start': np.float64(-37.152188838740585), 'E_reverse': np.float64(-0.027216778192551777), 'omega': np.float64(23.02050705566578), 'phase': 0, 'delta_E': np.float64(9.330216773192323), 'v': np.float64(1.0), 'Temp': 278, 'N_elec': 1, 'Surface_coverage': 1e-10, 'area': 0.07, 'Ru': np.float64(0.053591960547363454), 'CdlE2': -0.00071997893984, 'Cdl': np.float64(0.2459401704202494), 'k0': np.float64(742.8035787234024), 'CdlE1': -0.0035929255146, 'alpha': 0.4770065713, 'CdlE3': -5.3937670998e-06, 'E0_mean': -17.55609881985963, 'E0_std': 2.7137224476458734, 'gamma': 1.1283378372, 'E0': 0.0, 'theta': 0, 'phase_flag': 0, 'Marcus_flag': 0, 'tr': np.float64(37.12497206054804), 'cap_phase': 0},
    "15_Hz":{'E_start': np.float64(-37.15285703322782), 'E_reverse': np.float64(-0.026686123977569833), 'omega': np.float64(38.3661608491904), 'phase': 0, 'delta_E': np.float64(7.462119887777051), 'v': np.float64(1.0), 'Temp': 278, 'N_elec': 1, 'Surface_coverage': 1e-10, 'area': 0.07, 'CdlE3': -7.5051594548e-06, 'CdlE2': -0.00033463983363, 'E0_mean': -17.63877146804164, 'E0_std': 3.1937970053734026, 'alpha': 0.4006210586, 'gamma': 0.77951293716, 'CdlE1': -0.0016383464471, 'k0': np.float64(44.90143788942225), 'Ru': np.float64(0.012954696126217748), 'Cdl': np.float64(0.24667273415420504), 'E0': 0.0, 'theta': 0, 'phase_flag': 0, 'Marcus_flag': 0, 'tr': np.float64(37.12617090925025), 'cap_phase': 0},

}
amps=["80","280"]
for i in range(0,len(frequencies)):
    
    for j in range(1, len(amps)):
        amp=amps[j]
        full_loc=os.path.join(loc,amp)
        files=os.listdir(full_loc)
        filename=[file for file in files if frequencies[i] and amp+"_mV" in file][0]
        print(type(results_dict["FTACV"][frequencies[i]][amp]))
        for dictionary in [results_dict["FTACV"][frequencies[i]][str(amp)]]:
            dictionary["Temp"]=278
            dictionary["N_elec"]=1
            dictionary["Surface_coverage"]=1e-10
            dictionary["area"]=0.036

        slurm_class = sci.RunSingleExperimentMCMC(
            "FTACV",
            results_dict["FTACV"][frequencies[i]][amp],          
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
                            "omega":[0.8*dictionary["omega"], 1.2*dictionary["omega"]],
                            }

        slurm_class.GH_quadrature=True
        slurm_class.dispersion_bins=[30]
        slurm_class.Fourier_fitting=True
        slurm_class.top_hat_width=0.25
        slurm_class.Fourier_function="abs"
        slurm_class.Fourier_harmonics=list(range(3, 10))
        slurm_class.fixed_parameters={"phase":0}
        slurm_class.optim_list = ["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega", "alpha"]
        redimmed_dict=slurm_class.redimensionalise(nondims[frequencies[i]])
        print([float(redimmed_dict[x]) for x in slurm_class.optim_list])