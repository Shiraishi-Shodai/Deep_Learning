import math
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib import animation

"""
E = f(w) = 0.21*(w-2.34)**2+0.62　の最小値を求める問題
E' = 0.21*2*(w-2.34)
"""

def diff_func_p(w) :
    return np.array([0.21*2*(w[0]-2.34)])

def E(w):
    return 0.21*(w[0]-2.34)**2+0.62

E_history =[]
w_history =[]
epoch_history =[]
eta = 0.05 #安定的に定常に
#eta = 3.8
#eta = 4.2 #ばたつくが早い
#eta = 4.7 #ばたつくが早い
#eta = 4.75 #振動
eta = 0.1 #

w = np.array([8.5])
for epoch in range(1,501):
    grad = diff_func_p(w)
    w = w - eta * grad
    # 勾配が0　＝＞　ほんんど0ベクトル　＝＞　ベクトルの大きさがほとんど0
    print(f'epoch:{epoch} : w_1= {w[0]}\t,E={E(w)}\t,grad={grad}')
    E_history.append(E(w))
    w_history.append(w[0])
    epoch_history.append(epoch)

    if math.sqrt( sum([g**2 for g in grad]) )  < 0.001:
        print(f'エポック数{epoch}回にて収束しました')
        break
"""
plt.xlabel("epoch数")
plt.ylabel("E")
plt.plot(epoch_history,E_history,color="green")
plt.show()
"""
#fig = plt.figure()
fig, ax = plt.subplots(figsize=(10, 10))
ims = []

for i in range(len(epoch_history)):
    #plt.cla()
    x = np.linspace(-9,9,100)
    x0 = w_history[i] #18*(i/100)-9
    y = 2*0.21*(x0-2.34)*(x-x0)+0.21*(x0-2.34)**2+0.62
    y0 = 0.21*(x0-2.34)**2+0.62
    img = plt.plot(x,y,color="green") 
    #img = plt.scatter(np.array([x0]),np.array([y0]),color="green")
    plt.title(f"w ={i}回学習")
    plt.ylim(-10,10)
    plt.xlabel("w")
    plt.ylabel("E")
    ims.append(img) 

# 100枚のプロットを 100ms ごとに表示するアニメーション
ani = animation.ArtistAnimation(fig, ims, interval=100)
x = np.linspace(-9,9,100)
y2 = 0.21*(x-2.34)**2+0.62
plt.plot(x,y2,color="red")
plt.show()
ani.save("./k.gif", writer='pillow')