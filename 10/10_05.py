import math
import numpy as np
from sympy import *

# wの関数であるloss_funcからその値を最小にするwを求める。
def GetBestParam(loss_func):
    w0, w1= symbols('w0, w1')
    loss_f = loss_func(np.array([w0,w1]))
    print(diff(loss_f, w0))
    print(diff(loss_f, w1))
    sol = solve([diff(loss_f, w0), diff(loss_f, w1)],[w0,w1])
    print(sol)
    return [sol[w0],sol[w1]] 

def loss_func(w):
    return (w[0]-1)**2 + (w[1]-2)**2 -1

print(GetBestParam(loss_func))

