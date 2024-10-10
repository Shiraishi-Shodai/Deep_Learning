import numpy as np
def simple_neuron3(X,w,theta=0):
    u = X@w
    if u >= theta:
        return 1.0
    else:
        return 0.0

X = np.array([1.1,2.2])
w = np.array([0.2,0.5])
print(f'ニューロンの出力は{simple_neuron3(X,w)}')
