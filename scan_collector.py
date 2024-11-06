import numpy as np
import itertools
import os
list1=["E0_mean","E0_std","k0","gamma", "Ru","Cdl","CdlE1","CdlE2","CdlE3","omega","alpha","phase"]
combos=list(itertools.combinations(list1, 2))
for i in range(0, len(combos)):
    if "k0" in combos[i] and "Ru" in combos[i]:
        del_idx=i
del combos[del_idx]
for freq in ["3_Hz", "9_Hz", "15_Hz"]:
 files=os.listdir("tmp_scan_results/{0}".format(freq))
 for combo in combos:
  for i in range(0, 50):
   for file in files:
    if combo[0] in file and combo[1] in file and "scan_{0}_".format(str(i)) in file:
     if "Cdl" in combo:
      if "Cdl" not in file.split("_"):
       continue
     loaded=np.loadtxt(os.path.join("tmp_scan_results/{0}".format(freq), file))
     if i==0:
      savearray=loaded
     else:
      savearray=np.concatenate((savearray, loaded),axis=0)
     print(file, len(loaded), len(savearray), combo[0], combo[1])
     if len(savearray)>2500:
      raise ValueError(file)
  np.savetxt(os.path.join("scan_results","param_combos", freq+"_scan_results_{0}_{1}.txt".format(combo[0], combo[1])), np.array(savearray)) 
  del savearray
