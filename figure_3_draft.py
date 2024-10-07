import numpy as np
import Surface_confined_inference as sci
import matplotlib.pyplot as plt
import os
frequencies=[3,9,15]
file_names=["{0}_Hz_Full_table.txt".format(x) for x in frequencies]
for i in range(0, len(file_names)):
    values, titles=sci._utils.read_param_table(os.path.join("Fit_tables", file_names[i]),get_titles=True)
    best=np.array(values)
    scores=best[:,-1]
    idx=[range(0, 6)]#np.where(scores<(1.5*scores[0]))
    sizes=np.arange( 20-len(idx[0]),20)*2.5
    if i==0:
        det=sci._utils.det_subplots(len(best[0])-1)
        fig,axes=plt.subplots(*det)
    for j in range(0, len(best[0])-1):       
        ax=axes[j//det[1], j%det[1]]
        ax.scatter([frequencies[i]]*len(idx[0]), best[idx, j][0], s=sizes)
        if i==0:
            ax.set_ylabel(titles[j])
plt.show()
    
