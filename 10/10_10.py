import math
import numpy as np

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

def loss_func(w):
    print(w)
    return 3*w[0]**2 + w[1] -1

print(get_grad(loss_func,np.array([1.2,0.5]),h=0.0001))

