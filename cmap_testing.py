import numpy as np
z=np.logspace(0, 2, 50)
print(np.diff(z))
print(np.cumsum(np.flip(np.diff(z))))