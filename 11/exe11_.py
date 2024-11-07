"""
演習011 問（2）余力のある人向け
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
import math

df = pd.read_csv("exe11.csv")
X = df['x'].to_numpy().reshape(-1,1)
y = df['y'].to_numpy()

#数値微分の関数　f はベクトルvec_wの関数である。
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

# def get_grad(w):
#     global X,y
#     grad = np.zeros_like(w)
#     # np.mean((w[0]  + w[1]* np.sin(3.1416*X/5) + w[2]* X - y)**2)
#     grad[0] = np.mean(2 * (( w[0]  + w[1]* np.sin(3.1416*X/5) + w[2]* X) - y))
#     grad[1] = np.mean(2 * np.sin(3.1416*X/5) * (( w[0]  + w[1]* np.sin(3.1416*X/5) + w[2]* X) - y))
#     grad[2] = np.mean(2 * X * (( w[0]  + w[1]* np.sin(3.1416*X/5) + w[2]* X) - y))

#     return grad

#損失関数はw の関数
def loss_func(w):
    global X,y #お行儀が悪いけどグローバル変数
    # a1  + a2* sin(3.1416*x/5) + a3* x
    y_pred = w[0] + w[1] * np.sin(3.1416*X/5) + w[2]* X
    e = np.mean((y - y_pred.flatten()) ** 2)
    return e


w = np.zeros(3)
######ここで勾配降下法
max_learning = 1000
eta = 0.007
loss = []

for epoch in range(1,max_learning+1):
    grad = get_grad(loss_func, w)
    # grad = get_grad(w)
    w -= eta *grad
    #　以下で勾配降下法を止める条件
    if math.sqrt(sum([g**2 for g in grad])) < 0.0001:
        break
    print(f"epoch:{epoch} ,w={w}",end=" ")
    print(f"E={loss_func(w)}" )
    loss.append(loss_func(w))

plt.scatter(np.arange(max_learning),loss,color="green")
plt.savefig("loss.png")
plt.show()

y_pred = w[0] + w[1]* np.sin(3.1416*X/5) + w[2]* X
plt.plot(X,y_pred,color="red")#予測曲線
plt.scatter(X,y,color="blue") #与えられたCSVのデータ
plt.savefig("pred.png")
plt.show()
