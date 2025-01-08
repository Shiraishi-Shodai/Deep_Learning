import math
import numpy as np
import matplotlib.pyplot as plt

#2値分類のアイディア

data=np.array(
[
    [1,     1.1,    -1],
    [1,     3.9,    1],
    [1.5,   4.9,    1],#上が1
    [2,     1.9,    -1],
    [2.5,   5.5,    -1],
    [1.1,   1.9,    -1],
    [1.5,   2.5,    -1],

    [0.8,   5.4,    1],
    [1.2,   6.7,    1],
    [1.6,   5.4,    1],

    [1.7,   4.4,    1],
]
)
X = data[:,0]
y = data[:,1]
t= data[:,2]

#数値微分の関数
def diff_func_p(f,vec_w,h=0.0001) :
    grad = np.zeros_like(vec_w)
    for i in range(len(vec_w)):
        vec_i_org = vec_w[i]
        vec_w[i] = vec_i_org + h
        fh1=f(vec_w)
        vec_w[i] = vec_i_org - h
        fh2=f(vec_w)
        grad[i] = (fh1-fh2)/(2*h)
        vec_w[i] = vec_i_org
    return grad

#判定の誤差関数
def E(w):
    e =0
    for i in range(len(X)):
        e += max(0,-t[i]*(y[i]-(w[0]*X[i]+w[1]))) # この関数は間違った分類をすると0より大きい値を返す
        #e += max(0.02,-t[i]*(y[i]-(w[0]*X[i]**2+w[1]*X[i]+w[2])))
    
    return e/len(X) #+ 2.1* 1/np.sum((y-(w[0]*X**2+w[1]*X+w[2]))**2).mean()

#分類の予測の関数[x1,x2]の組を与えられたときに予測して　1 or -1を返す
def predict(x1,x2,f):
    label = 0
    if x2 - f(x1) >0 :
        label = 1
    elif x2 - f(x1) <0 :
        label = -1
    elif x2 - f(x1) ==0 :
        label = 0
    return label

# 以下勾配降下法で処理する
eta = 0.05
w = np.array([5.0,-1.2,5.5])
for epoch in range(1,2501):
    grad = diff_func_p(E,w,h=0.0001)
    w = w - eta * grad
    if math.sqrt( sum([g**2 for g in grad]) )  < 0.01:
        break
    print(f'epoch{epoch}: w_1= {w[0]} , w_2 = {w[1]},grad={grad},E={E(w)}')

print(w)
X_pred = np.linspace(0,2.5,20)
y_pred = w[0]*X_pred+w[1]
#y_pred = w[0]*X_pred**2 + w[1]*X_pred + w[2]
plt.xlabel("X1")
plt.ylabel("x2")
color_array = ['r' if c>0 else 'b' for c in t]
plt.scatter(X,y,c=color_array)
plt.plot(X_pred,y_pred,":",color="green")
plt.show()

