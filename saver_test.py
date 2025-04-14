import numpy as np
import Surface_confined_inference as sci
start= {'E_start': np.float64(-0.8900244598847762), 'E_reverse': np.float64(-0.0006520099910067856), 'omega': np.float64(9.104193706642272), 'phase': np.float64(5.682042161157082), 'delta_E': np.float64(0.22351633655321063), 'v': np.float64(0.059528212300390126)}
start["N_elec"]=1
start["Temp"]=298
start["area"]=0.036
start["Surface_coverage"]=1e-10

saver=sci.SingleExperiment("FTACV",start)
saver.optim_list=["E0","k0","gamma","Ru","Cdl","alpha"]
saver.save_class("test")
loader=sci.BaseExperiment.from_json("test.json", type="parallelsimulator")
print(loader.optim_list)
print(loader._internal_memory)