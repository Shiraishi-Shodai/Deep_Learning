import numpy as np
def simple_neuron(X,w,theta=0):
    u = X@w
    if u >= theta:
        return 1.0
    else:
        return 0.0

X = np.array([[1,1],[1,0],[0,1],[0,0]])
w = np.array([0.5,0.5])

for x_in in X:
    print(f'x={x_in}のとき',end=" : ")
    print(simple_neuron(x_in,w,1.0))