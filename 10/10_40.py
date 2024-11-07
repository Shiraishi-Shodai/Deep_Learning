import numpy as np
import matplotlib.pyplot as plt
def f(w):
    w1 = w[0]
    w2 = w[1]
    return (w1 - 1)**2 + w2**2 - 4*w2 + 3

x1 = np.linspace(-3, 8, 100)
x2 = np.linspace(-3, 8, 100)
x1_mesh, x2_mesh = np.meshgrid(x1, x2)
z = f(np.array((x1_mesh, x2_mesh)))

fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(x1, x2, z, levels=np.logspace(-0.3, 1.2, 10))
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_aspect('equal')
plt.show()
