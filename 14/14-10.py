import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

x_, y_ = np.arange(-2,2.01,0.01), np.arange(-2,2.01,0.01)
x, y = np.meshgrid(x_, y_)
z = (1-x)**2 + 100*(y-x**2)**2

levs = 10**np.arange(0., 3.5, 0.5)

plt.contour(x,y,z,norm=LogNorm(),levels=levs,cmap="viridis")
plt.colorbar()

plt.show()