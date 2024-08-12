import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
abspath="/home/henryll/Documents/Experimental_data/Nat/m4D2/"
#abspath="/home/userfs/h/hll537/Documents/Experimental_data/m4D2"
dataloc="m4D2_Data"
Blankloc="Blank"
experiments=["FTACV", "PSV"]
harmonic_range=list(range(0, 9))
plot_options=dict(zip(experiments, [
    {"hanning":True, "xaxis":"DC_potential", "plot_func":np.abs, "xlabel":"DC potential (V)", "ylabel":"Current ($\\mu$A)", "remove_xaxis":True,"filter_val":0.25},
    {"hanning":False, "xaxis":"potential", "plot_func":np.real, "xlabel":"Potential (V)", "ylabel":"Current ($\\mu$A)", "remove_xaxis":True}
]))
harmonics=len(harmonic_range)
for i in range(0, len(experiments)):
    if experiments[i]=="FTACV":
        grouping=["mV.txt","Hz"]
        labels=["mV", "Hz"]
        labeldict=dict(zip(grouping, labels))
        rowcol=dict(zip(grouping, [2, 4]))
    elif experiments[i]=="PSV":
        grouping=["Hz", "osc.txt"]
        labels=["Hz", "Oscillations"]
        labeldict=dict(zip(grouping, labels))
        rowcol=dict(zip(grouping, [4,1]))

    directory="/".join([abspath, dataloc, experiments[i]])
    files=os.listdir(directory)
    file_dict={}
    for j in range(0, len(files)):
        file_dict[files[j]]=np.loadtxt(directory+"/"+files[j])
    file_list=list(file_dict.keys())

    for j in range(0, len(grouping)):
        sortdict={}
        for m in range(0, len(file_list)):
            split=file_list[m].split("_")  
            
            idx=split.index(grouping[j])
            colkey=split[idx-1]
            plotkey=split[split.index(grouping[j-1])-1]

            if colkey in sortdict.keys():
                sortdict[colkey][plotkey]=file_dict[file_list[m]]
            else:
                sortdict[colkey]={plotkey:file_dict[file_list[m]]}
        
        plot_keys=list(sortdict.keys())
        plot_keys=[str(x) for x in sorted([int(x) for x in plot_keys])]
        fig,ax=plt.subplots()
        for m in range(0, len(plot_keys)):

            pk=plot_keys[m]
            trace_keys=[str(x) for x in sorted([int(x) for x in sortdict[pk]])]
            local_options=copy.deepcopy(plot_options[experiments[i]])
            
            for q in range(0, len(trace_keys)): 
                tk=trace_keys[q]
                datakey=tk+" "+labeldict[grouping[j-1]]+"_data"
                if experiments[i]=="FTACV":
                
                
                    time=sortdict[pk][tk][:,0]
                    current=sortdict[pk][tk][:,1]*1e6
                    potential=sortdict[pk][tk][:,2]
                    local_options[datakey]={"time":time, "potential":potential, "current":current, }
                    plt.plot(time, current)
                    plt.show()
                elif experiments[i]=="PSV":
                    time=sortdict[pk][tk][:,0]
                    current=sortdict[pk][tk][:,1]*1e6
                    potential=sortdict[pk][tk][:,2]
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
            axes[0].set_title(pk+" "+labeldict[grouping[j]])
            fig=plt.gcf()
            fig.set_size_inches(7, 9)
            plt.show()
            fig.savefig("Initial_plots/{2}/Grouped_by_{0}{1}.png".format(pk, labeldict[grouping[j]], experiments[i]), dpi=500)
                
           

    
