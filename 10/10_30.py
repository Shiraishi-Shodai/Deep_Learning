import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x = y = np.arange(-15, 15, 0.01)
X, Y = np.meshgrid(x, y)

z = (X - 1)**2 + Y**2 - 4*Y + 3

ax.plot_surface(X,Y,z, cmap='terrain')
plt.show()