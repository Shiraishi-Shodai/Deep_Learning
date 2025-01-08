import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# x,y軸の値を作成 # Rosenbrock関数
x = np.linspace(-2.0, 2.0, num=500)
y = np.linspace(-1.0, 3.0, num=500)

# x,y軸の値を作成 Himmelblau関数
x = np.linspace(-6.0, 6.0, num=500)
y = np.linspace(-6.0, 6.0, num=500)

# 格子状の点を作成
x_grid, y_grid = np.meshgrid(x, y)

# Rosenbrock関数
a = 1.0
b = 100.0
z_grid = b * (y_grid - x_grid**2)**2 + (a - x_grid)**2

#Himmelblau関数
z_grid = (x_grid**2 + y_grid -11 )**2 + (x_grid + y_grid**2 -7 )**2



fig = plt.figure(figsize=(12, 9)) 
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis') 
ax.set_xlabel('x') # x軸ラベル
ax.set_ylabel('y') # y軸ラベル
ax.set_zlabel('z') # z軸ラベル
plt.show()