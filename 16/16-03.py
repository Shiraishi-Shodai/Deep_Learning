
import math
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('test.csv')

grad_history =[]
epoch_history =[]
eta = 0.0001
w = np.array([4.0,2.7])

"""
#偏微分する関数（人間が手で計算した）
def get_grad(f,vec_w,h=0.0001) :
    w1 = vec_w[0]
    w2 = vec_w[1]
    return np.array([2*(w1 - 1),2*w2-4])
"""

def get_grad(f,vec_w,h=0.0001) :
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

def func_01(w):
    w1 = w[0]
    w2 = w[1]
    return (w1-1)**2 +w2**2 -4*w2 + 3


E_history =[]
epoch_history =[]
eta = 0.05
w = np.array([12.0,9.7])
for epoch in range(1,501):
    w = w - eta * get_grad(func_01,w,h=0.0001)
    # 勾配が0　＝＞　ほんんど0ベクトル　＝＞　ベクトルの大きさがほとんど0
    if math.sqrt( sum([i**2 for i in get_grad(func_01,w,h=0.0001)]) )  < 0.001:
        break
    print(str(epoch) +" th train : w_1= "+str(w[0]) + ' , w_2 = '+str(w[1]),end=" ")
    print('E='+str(func_01(w)) )
    E_history.append(func_01(w))
    epoch_history.append(epoch)

plt.xlabel("epoch")
plt.ylabel("E")
plt.plot(epoch_history,E_history,color="green")
plt.show()
