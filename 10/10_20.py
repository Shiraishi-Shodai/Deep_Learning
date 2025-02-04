import math
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

#数値微分の関数　f はベクトルvec_wの関数である。
# def get_grad(f,vec_w,h=0.0001) :
#     grad = np.zeros_like(vec_w)
#     for i in range(len(vec_w)):
#         vec_i_org = vec_w[i]
#         vec_w[i] = vec_i_org + h
#         fh1=f(vec_w)
#         vec_w[i] = vec_i_org - h
#         fh2=f(vec_w)
#         grad[i] = (fh1-fh2)/(2*h)
#         vec_w[i] = vec_i_org
#     return grad

# 解析的微分
def get_grad(f,vec_w,h=0.0001) :
    x = w[0]
    y = w[1]
    # (x-1)**2 +y**2 -4*y + 3 をxで微分 2(x - 1)
    # (x-1)**2 +y**2 -4*y + 3 をyで微分 2(y - ４)
    return np.array([2*(x - 1), 2*(y-4)]) # 損失関数を微分した導関数


# 損失関数 loss_func(ここでは活性化関数を恒等関数としている)
def loss_func(w):
    x = w[0]
    y = w[1]
    return (x-1)**2 +y**2 -4*y + 3

eta = 0.05
w = np.array([12.0,9.7])
for epoch in range(1,501):
    grad= get_grad(loss_func,w,h=0.0001)
    w = w - eta * grad
    if np.linalg.norm(grad,ord=2)  < 0.001:
        break
    print(f"epoch:{epoch} ,w={w}",end=" ")
    print(f"E={loss_func(w)}" )
    plt.scatter(epoch,loss_func(w),color="green")

plt.xlabel("学習回数")
plt.ylabel("関数Eの値")
plt.show()


