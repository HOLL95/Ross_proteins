import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
abspath="/home/henryll/Documents/Experimental_data/Nat/m4D2/"
#abspath="/home/userfs/h/hll537/Documents/Experimental_data/m4D2"
abspath="/home/henryll/Documents/Experimental_data/ForGDrive"

dataloc="Inference_round_2"
Blankloc="Blank"
experiments=["FTACV", "PSV"]
harmonic_range=list(range(0, 9))
plot_options=dict(zip(experiments, [
    {"hanning":True, "xaxis":"DC_potential", "plot_func":np.abs, "xlabel":"DC potential (V)", "ylabel":"Current ($\\mu$A)", "remove_xaxis":True,"filter_val":0.15},
    {"hanning":False, "xaxis":"potential", "plot_func":np.real, "xlabel":"Potential (V)", "ylabel":"Current ($\\mu$A)", "remove_xaxis":True}
]))
harmonics=len(harmonic_range)
for i in range(0, len(experiments)):
    if experiments[i]=="FTACV":
        directory="/".join([abspath, dataloc, experiments[i], "280"])
    elif experiments[i]=="PSV":
        directory="/".join([abspath, dataloc, experiments[i]])
    files=os.listdir(directory)
    file_dict={}
    for j in range(0, len(files)):
        file_dict[files[j]]=np.loadtxt(directory+"/"+files[j])
    file_list=list(file_dict.keys())

    
    sortdict={}
    for m in range(0, len(file_list)):
        split=file_list[m].split("_")  
        
        idx=split.index("Hz")
        colkey=split[idx-1]
        sortdict[colkey]=file_dict[file_list[m]]

        
    
    plot_keys=list(sortdict.keys())
    plot_keys=[str(x) for x in sorted([int(x) for x in plot_keys])]
    
    
    for q in range(0, len(plot_keys)): 
        pk=plot_keys[q]
        datakey=pk+" Hz_data"
        local_options=copy.deepcopy(plot_options[experiments[i]])
        if experiments[i]=="FTACV":
            
            time=sortdict[pk][:,0]
            current=sortdict[pk][:,1]*1e6
            potential=sortdict[pk][:,2]
            local_options[datakey]={"time":time, "potential":potential, "current":current, }
        elif experiments[i]=="PSV":
            
            time=sortdict[pk][:,0]
            current=sortdict[pk][:,1]*1e6
            potential=sortdict[pk][:,2]
            freq=sci.get_frequency(time, current)
            get_rid=5/freq
            idx=np.where(time>get_rid)
            time=time[idx]
            potential=potential[idx]
            current=current[idx]
            local_options[datakey]={"time":time, "potential":potential, "current":current, "harmonics":list(range(3, 11))}
        
        
        local_options["h_num"]=True
        

    
        local_options["legend"]={"loc":"upper right", "ncol":2, "bbox_to_anchor":[1, 2], "frameon":False}
        axes=sci.plot.plot_harmonics(**local_options)
        #axes[0].set_title(pk+" "+labeldict[grouping[j]])
        fig=plt.gcf()
        fig.set_size_inches(7, 9)
        plt.show()
        #fig.savefig("Initial_plots/Set2/{0}/Initial_plot.png".format(experiments[i]), dpi=500)
                
           

    
