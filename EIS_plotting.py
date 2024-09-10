import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt("/home/henryll/Documents/Experimental_data/Nat/m4D2_set2/EIS/EIS_m4D2_PGE_0.1_-_100_kHz_-440_mV_5_deg.csv", skiprows=1, delimiter=",")
truncate=15
freq=np.flip(data[truncate:,0])
real=np.flip(data[truncate:,1])
imag=np.flip(data[truncate:,2])
simulator=sci.SimpleSurfaceCircuit()
test_freq=np.logspace(0, 5, 100)


spectra=np.column_stack((real, imag))
sci.plot.bode(spectra,freq)
plt.show()
Ru=sci.infer.EIS_solution_resistance(freq, spectra, plot_results=True,Cdl="Qdl",Cf="Qf")#  Cf="Qf"
print(Ru)