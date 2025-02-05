import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib

def ackley_func(w):
    x = w[0]
    y = w[1]
    pi = 3.1415926535
    e = 2.718281
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 *
    pi * x)+np.cos(2 * pi * y))) + e + 20

# 数値微分の関数 f はベクトルvec_wの関数である。
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

epochs = 1000
eta =  0.001
loss_history = []
weight_history = []
w = np.array([2.41893862, 2.51036339])
v = np.zeros_like(w)  # モーメンタム項の初期化
beta = 0.97115
h0 = 0.00001
h =np.array([h0,h0])

for epoch in range(1, epochs + 1):
    grad = get_grad(ackley_func, w, h=0.0001)

    # モーメンタム
    v = beta * v - eta * grad
    w += v

# AdaGrad
    # h = h + grad **2
    # w = w - eta * grad / np.sqrt(h)

    weight_history.append([w[0], w[1], ackley_func(w)])
    if epoch % 10 == 0:
        print(f"epoch:{epoch} ,w={w}", end=" ")
        print(f"E={ackley_func(w)}")
        print(f"勾配 {np.linalg.norm(w, ord=2)}")
    
    if np.linalg.norm(w, ord=2) < 0.0001:
        break


weight_history = np.array(weight_history)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('損失関数')
w1 = np.linspace(-2, 3, 100)
w2 = np.linspace(-2, 3, 100)
ww1, ww2 = np.meshgrid(w1, w2)  
w3=np.zeros((len(w1),len(w2)))

for i in range(0,len(w2)):
    for j in range(0,len(w1)):
        w3[i][j]=ackley_func(np.array([ww1[i][j],ww2[i][j]]))
ax.plot_wireframe(ww1, ww2, w3, color='blue',linewidth=0.3)
ax.scatter(weight_history[:,0], weight_history[:,1], weight_history[:,2], color='red')
# plt.savefig("rosenbrock.png")
plt.show()