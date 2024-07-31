import numpy as np
import matplotlib.pyplot as plt
t=np.linspace(0, 100, 100000)
phase=3*np.pi/2

phase2=np.sin(0.1*t+0.5)
a=np.sin(2*np.pi*t+phase2)
b=np.sin((2*np.pi+0.1)*t+0.5)
plt.plot(t, a)
plt.plot(t, b)
plt.plot(t, a-b)
plt.show()