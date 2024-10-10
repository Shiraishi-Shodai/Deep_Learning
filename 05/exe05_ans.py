import numpy as np
import pandas as pd
from random import uniform

def neuron_output(x1,x2,w1,w2,w3,theta=0.5):
    u = w1*x1 +w2*x2+w3*x1*x2
    z =  1 if u> theta else 0
    return z


for _ in range(0,10000):
    w1 = uniform(-2,2)
    w2 = uniform(-2,2)
    w3 =0
    theta = uniform(-1,2)
    if neuron_output(1,1,w1,w2,w3,theta) ==0 and neuron_output(0,0,w1,w2,w3,theta)==1 and  neuron_output(1,0,w1,w2,w3,theta) ==1 and neuron_output(0,1,w1,w2,w3,theta) ==1 :
        print("OK!!")
        print(f"{w1:.3f},{w2:.3f},{theta:.3f} => ",end="\t")
        print(f"{neuron_output(1,1,w1,w2,w3,theta)}," ,end=" ")
        print(f"{neuron_output(1,0,w1,w2,w3,theta)}," ,end=" ")
        print(f"{neuron_output(0,1,w1,w2,w3,theta)}," ,end=" ")
        print(f"{neuron_output(0,0,w1,w2,w3,theta)},")
        break
    else:
        pass#print("NG")
print(f"-----------------------")

for _ in range(0,10000):
    w1 = uniform(-2,2)
    w2 = uniform(-2,2)
    w3 = uniform(-1,2)
    if neuron_output(1,1,w1,w2,w3,0.55) ==0 and neuron_output(0,0,w1,w2,w3,0.55)==0 and  neuron_output(1,0,w1,w2,w3,0.55) ==1 and neuron_output(0,1,w1,w2,w3,0.55) ==1 :
        print("OK!!")
        print(f"{w1:.3f},{w2:.3f},{w3:.3f} => ",end="\t")
        print(f"{neuron_output(1,1,w1,w2,w3,0.55)}," ,end=" ")
        print(f"{neuron_output(1,0,w1,w2,w3,0.55)}," ,end=" ")
        print(f"{neuron_output(0,1,w1,w2,w3,0.55)}" ,end=" ")
        print(f"{neuron_output(0,0,w1,w2,w3,0.55)},")
        break
    else:
        pass#print("NG")