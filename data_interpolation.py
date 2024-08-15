import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from scipy.optimize import fmin
abspath="/home/henryll/Documents/Experimental_data/Nat/m4D2/"
#abspath="/home/userfs/h/hll537/Documents/Experimental_data/m4D2"
dataloc="m4D2_Data"
Blankloc="Blank"
experiments=["FTACV", "PSV"]
results_dict={"FTACV":{"80":{}, "250":{}}, "PSV":{"100":{}}}
results_dict['FTACV']['80']['3']={'E_start': np.float64(-0.8006475758896721), 'E_reverse': np.float64(-0.0005643140832054527), 'omega': np.float64(2.7893933220163056), 'phase': np.float64(6.0963661768311965), 'delta_E': np.float64(0.07773033393989036), 'v': np.float64(0.0553759383170455), 'phase_delta_E': np.float64(0.017041489489115325), 'phase_omega': np.float64(0.1), 'phase_phase': np.float64(4.450487852784928)}
results_dict['FTACV']['80']['9']={'E_start': np.float64(-0.8006117990463852), 'E_reverse': np.float64(-0.0005378379312415138), 'omega': np.float64(8.368818213300784), 'phase': np.float64(5.724833637340166), 'delta_E': np.float64(0.06684109310228693), 'v': np.float64(0.05537358012592774), 'phase_delta_E': np.float64(0.0008055595726608189), 'phase_omega': np.float64(395.718786830266), 'phase_phase': np.float64(1.3981535550876787)}
results_dict['FTACV']['80']['15']={'E_start': np.float64(-0.8006003712479526), 'E_reverse': np.float64(-0.0005511926856689442), 'omega': np.float64(13.948051829113664), 'phase': np.float64(5.486141199502961), 'delta_E': np.float64(0.054767122705694136), 'v': np.float64(0.055374442627008344), 'phase_delta_E': np.float64(0.0006014194061334521), 'phase_omega': np.float64(224.77525169930774), 'phase_phase': np.float64(0.8121639027303783)}
results_dict['FTACV']['80']['21']={'E_start': np.float64(-0.8006001654780444), 'E_reverse': np.float64(-0.0005431127034229988), 'omega': np.float64(19.527270389888812), 'phase': np.float64(5.326737618246792), 'delta_E': np.float64(0.044884733179813445), 'v': np.float64(0.055374519015733634), 'phase_delta_E': np.float64(0.0014226254150879747), 'phase_omega': np.float64(78.44189968465425), 'phase_phase': np.float64(1.342048627726686)}
results_dict['FTACV']['250']['3']={'E_start': np.float64(-0.8004676428274355), 'E_reverse': np.float64(-0.0005662867260471938), 'omega': np.float64(2.7895996695937977), 'phase': np.float64(6.06881610886128), 'delta_E': np.float64(0.2425898161749251), 'v': np.float64(0.055362889245101125), 'phase_delta_E': np.float64(0.000896632559170385), 'phase_omega': np.float64(334.6748600982876), 'phase_phase': np.float64(1.5545034546151042)}
results_dict['FTACV']['250']['9']={'E_start': np.float64(-0.8004057616703748), 'E_reverse': np.float64(-0.0006082039098393999), 'omega': np.float64(8.36881360008673), 'phase': np.float64(5.717100130351908), 'delta_E': np.float64(0.20733790262971458), 'v': np.float64(0.05535907812595028), 'phase_delta_E': np.float64(-0.0004392453858779177), 'phase_omega': np.float64(241.2475829845014), 'phase_phase': np.float64(4.017544165276198)}
results_dict['FTACV']['250']['15']={'E_start': np.float64(-0.800374111924057), 'E_reverse': np.float64(-0.0006443841911099035), 'omega': np.float64(13.948046784482719), 'phase': np.float64(5.479088647354352), 'delta_E': np.float64(0.16931085920803546), 'v': np.float64(0.055355430054032324), 'phase_delta_E': np.float64(-0.0003732095390789336), 'phase_omega': np.float64(323.27848934212875), 'phase_phase': np.float64(3.7644641400758823)}
results_dict['FTACV']['250']['21']={'E_start': np.float64(-0.8004321886491068), 'E_reverse': np.float64(-0.0006032864694693885), 'omega': np.float64(19.527278389738022), 'phase': np.float64(5.320629887475525), 'delta_E': np.float64(0.1387153181670516), 'v': np.float64(0.055360894910183214), 'phase_delta_E': np.float64(0.0011065728791195273), 'phase_omega': np.float64(64.53511713089918), 'phase_phase': np.float64(1.3440760913658238)}
results_dict['PSV']['100']['3']={'Edc': np.float64(-0.400924233987332), 'omega': np.float64(2.9977559238552307), 'phase': np.float64(4.480463366514855), 'delta_E': np.float64(0.3867158021828757), 'phase_delta_E': np.float64(-0.005642700425637415), 'phase_omega': np.float64(360.72205406076256), 'phase_phase': np.float64(0.8707021064468491)}
results_dict['PSV']['100']['9']={'Edc': np.float64(-0.4002382809977385), 'omega': np.float64(8.993290590781472), 'phase': np.float64(4.109126055232471), 'delta_E': np.float64(0.3242408399795645), 'phase_delta_E': np.float64(-0.0018387849767278475), 'phase_omega': np.float64(226.04841410221107), 'phase_phase': np.float64(3.186375074140818)}
results_dict['PSV']['100']['15']={'Edc': np.float64(-0.40027596880288496), 'omega': np.float64(14.988781419399213), 'phase': np.float64(3.8673457767858714), 'delta_E': np.float64(0.25961556953091686), 'phase_delta_E': np.float64(0.0011854343898596298), 'phase_omega': np.float64(92.76606968064118), 'phase_phase': np.float64(1.2097269862812539)}
results_dict['PSV']['100']['21']={'Edc': np.float64(-0.4002616829802341), 'omega': np.float64(20.98454610332337), 'phase': np.float64(3.7133684339751927), 'delta_E': np.float64(0.21096337362617562), 'phase_delta_E': np.float64(-0.0003865370447724281), 'phase_omega': np.float64(2.7572035717779357), 'phase_phase': np.float64(0.1714948137654772)}
row_keys=["80","250","100"]

saved="/home/henryll/Documents/Experimental_data/ForGDrive/Interpolated/"
for i in range(0, len(experiments)):
    if experiments[i]=="FTACV":
        grouping=["mV.txt", "Hz"]
        rows=2
        cols=4
        
    elif experiments[i]=="PSV":
        grouping=["osc.txt", "Hz"]
        rows=2
        cols=2
    
    directory="/".join([abspath, dataloc, experiments[i]])
    files=os.listdir(directory)
    file_dict={}
    for j in range(0, len(files)):
        data=np.loadtxt(directory+"/"+files[j])
        time=data[:,0]
        potential=data[:,2]
        current=data[:,1]
        freq=sci.get_frequency(time, current)
        end_time=35/freq
        start_time=0
        if experiments[i]=="FTACV":
            chopped_time=time
        else:
            chopped_time=time[np.where((time<end_time) & (time>=start_time))]
            
        interped_time=np.linspace(chopped_time[0], chopped_time[-1], len(chopped_time))
        interped_potential=np.interp(interped_time, time, potential)
        interped_current=np.interp(interped_time, time, current)
        if experiments[i]=="PSV":
            plt.plot(interped_potential, interped_current)
            plt.show()
        np.savetxt("/".join([saved, experiments[i], files[j]]), np.column_stack((interped_time, interped_current, interped_potential)))

   
              
           

    
