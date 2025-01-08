import math
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
#数値微分の関数　f はベクトルvec_wの関数である。
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
    x = w[0]
    y = w[1]
    return (x-1)**2 +y**2 -4*y + 3

eta = 0.05
grad_p=[]
w_history=[]
w = np.array([12.0,9.7])
for epoch in range(1,501):
    grad= get_grad(func_01,w,h=0.0001)
    w = w - eta * grad
    w_history.append([w[0],w[1],func_01(w)])# 10_20.pyに追加。
    if np.linalg.norm(grad,ord=2)  < 0.001:
        break
    print(f"epoch:{epoch} ,w={w}",end=" ")
    print(f"E={func_01(w)}" )
    plt.scatter(epoch,func_01(w),color="green")

plt.xlabel("学習回数")
plt.ylabel("関数Eの値")
plt.show()

#### 10_20.pyに以下を追加。

w_history= np.array(w_history)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('w1')
ax.set_ylabel('w1')
ax.set_zlabel('E')
w1 = np.linspace(-10, 11, 100)
w2 = np.linspace(-10, 11, 100)
w1, w2 = np.meshgrid(w1, w2)  
#E =(w1-1)**2 +w2**2 -4*w2 + 3
E = func_01([w1,w2])
print(E.shape)
ax.plot_wireframe(w1, w2, E, color='blue',linewidth=0.3)
ax.scatter(w_history[:,0],w_history[:,1], w_history[:,2], color='red')
plt.show()