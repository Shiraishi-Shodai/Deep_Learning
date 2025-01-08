from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def ackley_func(x, y):
 return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * 
  pi * x)+cos(2 * pi * y))) + e + 20

# x,y軸の値を作成 # Rosenbrock関数
x = np.linspace(-2.0, 2.0, num=500)
y = np.linspace(-1.0, 3.0, num=500)


# 格子状の点を作成
x_grid, y_grid = np.meshgrid(x, y)

# ackley関数
z_grid = ackley_func(x_grid,y_grid)


fig = plt.figure(figsize=(12, 9)) 
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis') 
ax.set_xlabel('x') # x軸ラベル
ax.set_ylabel('y') # y軸ラベル
ax.set_zlabel('z') # z軸ラベル
plt.show()