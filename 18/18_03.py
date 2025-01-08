import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random as rd
from math import *
from sklearn.datasets import make_moons

def plot_boundary(vec_w, X, Y, target, xlabel, ylabel):
    cmap_dots = ListedColormap([ "#1f77b4", "#ff7f0e", "#2ca02c"])
    cmap_fills = ListedColormap([ "#c6dcec", "#ffdec2", "#cae7ca"])
    #ステップ関数
    def step_func(x,theta=0.5):
        if x >= theta :
            return 1.0
        else :
            return 0.0

    plt.figure(figsize=(5, 5))
    vfunc = np.vectorize(step_func)
    if 1:
        XX, YY = np.meshgrid(
            np.linspace(X.min()-1, X.max()+1, 200),
            np.linspace(Y.min()-1, Y.max()+1, 200))
        pred = vfunc(vec_w[0]*XX + vec_w[1]*YY+vec_w[2])
        #pred = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        plt.pcolormesh(XX, YY, pred, cmap=cmap_fills, shading="auto")
        plt.contour(XX, YY, pred, colors="gray") 
    plt.scatter(X, Y, c=target, cmap=cmap_dots)
    #print("-------------------------------")
    #print(pred)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

"""
勾配降下法から学習則を導出した場合
"""
#-----------------------------------------------
# test02.csvの場合

df = pd.read_csv('sample_2d.csv')
group01 = df[df['class']<0.1]
group02 = df[df['class']>0.1]
group01 =group01.values 
group02 =group02.values  
#-----------------------------------------------
# 楕円の場合
"""
def make_cir(center_x,center_y,a,b,num,clf_label=0):
    points = []
    for i in range(0,num):
        theta = rd.uniform(0,3.14159*2)
        x = a * cos(theta) * rd.random() + center_x
        y = b * sin(theta) * rd.random() + center_y
        points.append([x,y,clf_label])
    return np.array(points)

group01 = make_cir(4.0,1.0,2.0,1.5,200,0)
group02 = make_cir(0.7,2.0,1.8,1.1,200,1)

group01 = make_cir(4.0,1.0,2.0,1.5,200,0)
group02 = make_cir(2.7,3.0,1.8,1.1,200,1)
"""
#-----------------------------------------------
#moonの場合
"""
X,y = make_moons(n_samples=100)
X0 = X[y==0]
X0_y = np.zeros(len(X0)) 
group01 = np.concatenate([X0,X0_y.reshape(-1,1)],1)
X1 = X[y==1]
X1_y = np.ones(len(X1)) 
group02 = np.concatenate([X1,X1_y.reshape(-1,1)],1)
"""

data_XY = np.concatenate([group01,group02])
#print(data_XY)

def activation_function(x):
    return x

#損失関数
def Error_func(vec_w):
    global data_XY #グローバル変数。イケテないが
    w1 = vec_w[0]
    w2 = vec_w[1]
    w3 = vec_w[2]
    E=0
    for li in data_XY:
        x1 = li[0]
        x2 = li[1]
        t = li[2]
        E += (t-activation_function(w1*x1+w2*w2+w3))**2
    
    return E/(len(data_XY)*1000)

eta = 0.001 #学習係数
vec_w=np.array([3.2,0.9,3.0]) #重みの初期値

for epoc in range(1,1000):
    for li in data_XY:
        x1 = li[0]
        x2 = li[1]
        t = li[2]
        vec_w = vec_w - eta *(vec_w[0]*x1 +vec_w[1]*x2 +vec_w[2]-t)* np.array([x1,x2,1.0])
    
    print(str(epoc) +" th train : w_1= "+str(vec_w[0]) + ' , w_2 = '+str(vec_w[1]),end=" ")
    print("E="+str(Error_func(vec_w)) )
	
print(Error_func([vec_w[0],vec_w[1],vec_w[2]]))

for arry in data_XY:
    predict = vec_w[0]*arry[0] + vec_w[1]*arry[1]+vec_w[2]
    print(f"x1={arry[0]}\t, x2={arry[1]}\t, clf_label={arry[2]},predict={predict}")
    #print(f"x1={arry[0]}\t, x2={arry[1]}\t, clf_label={arry[2]}, predict={step_func(predict,0.5)}")


print(f"weight is finally w0={vec_w[0]}\t, w1={vec_w[1]}\t, w2={vec_w[2]}")
p = np.linspace( -2, 6, 100)   # linspace(min, max, N) で範囲 min から max を N 分割します
q = 1/vec_w[1]*(-p*vec_w[0] - vec_w[2]+0.5) # step_func(predict,0.5) の閾値0.5  を入れる。

plot_boundary(vec_w, data_XY[:,0],data_XY[:,1], data_XY[:,2], "X", "Y")
#plot_boundary(vec_w, df['X'].values,df['y'].values, df['label'].values, "X", "Y")