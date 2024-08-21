import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt("/home/henryll/Documents/Experimental_data/Ella/Chrono/blank chrono_2.txt")
time=data[:,0]
current=data[:,1]

diff=np.abs(np.diff(current))
time_diff=np.mean(np.diff(time))
mean=np.mean(diff)
ratio=diff/mean
ratio_cutoff=50

peak_loc=np.where(ratio>ratio_cutoff)
approx_time_between_pulses=20

peak_times=time[1:][peak_loc]
time_locations=[peak_times[0]]

for i in range(0, len(peak_times)):
    current_time=time_locations[-1]
    if peak_times[i]<(current_time+approx_time_between_pulses):
        continue
    else:
        time_locations.append(peak_times[i])
        print(time_locations)
time_locations=np.array(time_locations)
peaks=[current[np.where(time==x)][0] for x in time_locations]
print(peaks)
plt.plot(time,current)
plt.scatter(time_locations, peaks,color="red")
for i in range(0, len(time_locations)):
    if i==len(time_locations)-1:
        second_idx=time[-1]
    else:
        second_idx=time_locations[i+1]

    exclude_timestep=20
    data_chunk=np.where((time>(time_locations[i]+(exclude_timestep*time_diff))) & (time<second_idx-(exclude_timestep*time_diff)))
    
    mean=np.mean(current[data_chunk])
    plot_current=current[data_chunk]
    plt.plot(time[data_chunk], np.ones(len(plot_current))*mean, color="black", linestyle="--")
    plt.plot(time[data_chunk], plot_current)
plt.show()