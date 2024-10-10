import numpy as np
def simple_neuron2(X,w,theta=0):
    u = 0
    for i in range(0,len(X)):
        for j in range(0,len(w)):
            u += X[i]*w[j]

    if u >= theta:
        return 1.0
    else:
        return 0.0

X = np.array([1.1,2.2])
w = np.array([0.2,0.5])
print(f'ニューロンの出力は{simple_neuron2(X,w)}')
