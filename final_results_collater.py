import numpy as np
import sys
import os
final_save_dict={}
file_loc=sys.argv[1]
file_handle=sys.argv[2]
files=os.listdir(file_loc)
for file in file_loc:
    if file_handle in file_loc:
        dictionary=np.load(os.path.join(file_loc, file), allow_pickle=True).item()
        if dictionary["key"] not in final_save_dict:
            final_save_dict[dictionary["key"]]=[dictionary]
        else:
            final_save_dict[dictionary["key"]].append(dictionary)
        os.remove(os.path.join(file_loc, file))
np.save("frontier_results/simulation_values", final_save_dict)