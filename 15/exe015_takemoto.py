import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def ackley_func(w):
    x, y = w[0], w[1]
    pi = 3.1415926535
    e = 2.718281
    return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))-np.exp(0.5 * (np.cos(2 * pi * x) + np.cos(2 * pi * y))) + e + 20


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

# 学習率とモーメンタム係数の設定
eta = 0.05
beta = 0.9
w = np.array([1.5, 1.5])  # 初期値
v = np.zeros_like(w)  # モーメンタム項の初期化
w_history = []

h0 = 0.00001
h = np.array([h0, h0])

for epoch in range(1, 1001):
    grad = get_grad(ackley_func, w, h=0.0001)
    
    # AdaGradにモーメンタム
    h += grad**2
    v = beta * v - eta * grad / np.sqrt(h)
    w = w + v
    
    w_history.append([w[0],w[1],ackley_func(w)])
    if np.linalg.norm(grad, ord=2) < 0.001:
        break
    print(f"epoch:{epoch} ,w={w}", end=" ")
    print(f"E={ackley_func(w)}")
    plt.scatter(epoch, ackley_func(w), color="green")

plt.xlabel("学習回数")
plt.ylabel("関数Eの値")
plt.show()

w_history = np.array(w_history)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('損失関数')
w1 = np.linspace(-2, 2, 100)
w2 = np.linspace(-2, 2, 100)
ww1, ww2 = np.meshgrid(w1, w2)
w3 = np.zeros((len(w1), len(w2)))

for i in range(len(w2)):
    for j in range(len(w1)):
        w3[i][j] = ackley_func(np.array([ww1[i][j], ww2[i][j]]))
ax.plot_wireframe(ww1, ww2, w3, color='blue', linewidth=0.3)
ax.scatter(w_history[:, 0], w_history[:, 1], w_history[:, 2], color='red')
plt.savefig("rosenbrock.png")
plt.show()
