import math
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
"""
ackley関数を
"""

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

def loss_func(w):
    x = w[0]
    y = w[1]
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + e + 20

def find_minE(eta):
    # 学習率とモーメンタム係数の設定
    #eta = 0.001#0.001
    beta = 0.95
    #w = np.array([-0.8, 1.7])  # 初期値
    w = np.array([-0.1,2.7]) 
    v = np.zeros_like(w)  # モーメンタム項の初期化
    w_history=[]

    h0 = 0.00001
    h =np.array([h0,h0])
        
    for epoch in range(1, 1001):
        grad = get_grad(loss_func, w, h=0.0001)
        # Momemtum
        #eta= 0.009
        #v = beta * v - eta * grad  # モーメンタムの更新
        #w = w + v  # パラメータの更新
        
        # AdaGrad
        #eta =0.64
        #eta= 0.73
        h = h + grad**2
        w = w - eta * grad /np.sqrt(h)
        
        #Normal
        #w = w- eta*grad
        
        w_history.append([w[0],w[1],loss_func(w)])
        if np.linalg.norm(grad, ord=2) < 0.001:
            break
        #print(f"epoch:{epoch} ,w={w}", end=" ")
        #print(f"E={loss_func(w)}")
        #plt.scatter(epoch, loss_func(w), color="green")
    print(loss_func(w))
    return loss_func(w)
min_E= 1000000000
for eta in arange(0.009,1.2,0.001):
    E = find_minE(eta)
    if E < min_E:
        min_E =E
        min_set =[min_E,eta]
    print(f"eta ={eta},E={E}")

print(f"min_E is {min_E},{min_set}")
"""
plt.xlabel("学習回数")
plt.ylabel("関数Eの値")
plt.show()

w_history = np.array(w_history)
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
        w3[i][j]=loss_func(np.array([ww1[i][j],ww2[i][j]]))
ax.plot_wireframe(ww1, ww2, w3, color='blue',linewidth=0.3)
ax.scatter(w_history[:,0], w_history[:,1], w_history[:,2], color='red')
plt.savefig("rosenbrock.png")
plt.show()
"""