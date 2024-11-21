import math
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

def get_grad(f, vec_w, h=0.0001):
    grad = np.zeros_like(vec_w)
    for i in range(len(vec_w)):
        vec_i_org = vec_w[i]
        vec_w[i] = vec_i_org + h
        fh1 = f(vec_w)
        vec_w[i] = vec_i_org - h
        fh2 = f(vec_w)
        grad[i] = (fh1 - fh2) / (2 * h)
        vec_w[i] = vec_i_org
    
    return grad

def loss_fn(w):
    x = w[0]
    y = w[1]
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * 
  pi * x)+cos(2 * pi * y))) + e + 20

rng = np.random.default_rng()
w = rng.uniform(1, 5, 2)
w = np.array([2.41893862, 2.51036339])

eta = 0.001
epoch_num = 1000
w_history = []

# モーメンタム用の項
beta = 0.97115 # 0.00429
v = np.zeros_like(w)

# AdaGrad用の項
h0 = 0.00001
h =np.array([h0,h0])

for epoch in range(1, epoch_num + 1):

    # Normal
    # grad = get_grad(loss_fn, w, h=0.0001)
    # w =- eta * grad

    # AdaGrad
    # grad = get_grad(loss_fn, w, h=0.0001)
    # h = h + grad **2
    # w = w - eta * grad / np.sqrt(h)

    # Mometum
    grad = get_grad(loss_fn, w)
    v = beta * v - eta * grad
    w = w + v

    w_history.append([w[0], w[1], loss_fn(w)])
    # 終了条件
    if np.linalg.norm(grad, ord=2) < 0.001:
        break
    print(f"epoch:{epoch} ,w={w}", end=" ")
    print(f"E={loss_fn(w)}")

w_history = np.array(w_history)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("損失関数")

# x,y軸の値を作成 # Rosenbrock関数
w1 = np.linspace(-2.0, 3.0, num=500)
w2 = np.linspace(-2.0, 3.0, num=500)
ww1, ww2 = np.meshgrid(w1, w2)
w3 = np.zeros((len(w1), len(w2)))

for i in range(0,len(w2)):
    for j in range(0,len(w1)):
        w3[i][j]=loss_fn(np.array([ww1[i][j],ww2[i][j]]))
ax.plot_wireframe(ww1, ww2, w3, color='blue',linewidth=0.3)
ax.scatter(w_history[:,0], w_history[:,1], w_history[:,2], color='red')
plt.savefig("ackley.png")
plt.show()

