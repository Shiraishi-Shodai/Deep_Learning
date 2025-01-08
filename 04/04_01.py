def simple_neuron(x1,x2,w1,w2,theta=0):
    u = x1*w1+x2*w2
    if u >= theta:
        return 1.0
    else:
        return 0.0

X = [1.1,2.2]
w = [0.2,0.5]
print(f'ニューロンの出力は{simple_neuron(X[0],X[1],w[0],w[1])}')
#print(f'ニューロンの出力は{simple_neuron(1.1,2.2,0.2,0.5)}')
