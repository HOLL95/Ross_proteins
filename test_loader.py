import numpy as np
import os
import sys
p_dict=np.load(os.path.join(sys.argv[1], "saved_parameters.npy"), allow_pickle=True).item()
combo_keys=list(p_dict.keys())
points=50
keylen=len(combo_keys)
for i in range(0, 750):
 keydx=i//50
 idx=i%50
 front_list=p_dict[combo_keys[keydx]]
 if len(front_list)>0: 
  plist=front_list[idx]["parameters"]
  print(keydx, idx)

 

