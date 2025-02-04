import math
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def get_grad(f, vec_w, h=0.0001):
    grad = np.zeros_like(vec_w)
    for i in range(len(vec_w)):
        vec_i_org = vec_w[i]
        vec_w[i] = vec_i_org + h
        fh1 = f(vec_w)
        vec_w[i] = vec_i_org - h
        fh2 = f(vec_w)
        grad[i] = (fh1 - fh2) / (2*h)
        vec_w[i] = vec_i_org
    return grad

def get_grad2(vec_w):
    x = vec_w[0]
    y = vec_w[1]
    return np.array([2*(x - 1), 2 * (y - 4)])

def loss_func(vec_w):
    x = vec_w[0]
    y = vec_w[1]
    return (x - 1)**2 + y**2 - 4*y + 3

eta = 0.05
vec_w = np.array([12.0, 9.7])
w_history = []

for epoch in range(1, 501):
    # grad = get_grad(loss_func, vec_w, h=0.0001)
    grad = get_grad2(vec_w)
    vec_w = vec_w - eta * grad
    w_history.append([vec_w[0], vec_w[1], loss_func(vec_w)])
    print(f"勾配: {math.sqrt(sum([g**2 for g in grad]))}")
    if math.sqrt(sum([g**2 for g in grad])) < 0.0001:
        break
    print(f"epoch:{epoch} ,w={vec_w}",end=" ")
    print(f"E={loss_func(vec_w)}" )
    # plt.scatter(epoch,loss_func(vec_w),color="green")

w_history = np.array(w_history)
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("E")
w1 = np.linspace(-10, 11, 100)
w2 = np.linspace(-10, 11, 100)
w1, w2 = np.meshgrid(w1, w2)  
#E =(w1-1)**2 +w2**2 -4*w2 + 3
E = loss_func([w1,w2])
# print(E.shape) // (100, 100)
ax.plot_wireframe(w1, w2, E, color='blue',linewidth=0.3)
ax.scatter(w_history[:,0],w_history[:,1], w_history[:,2], color='red')
plt.show()

print(w_history.shape)