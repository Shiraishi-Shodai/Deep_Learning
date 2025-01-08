import numpy as np
from random import uniform

def neuron_output(x1, x2, w1, w2, theta1, theta2):
    u1 = x1 * w1 + x2 * w2
    u2 = x1 * 1.0 + x2 * 1.0

    v1 = u1
    v2 = 1 if u2 > theta1 else 0
    z = 1 if (v1 * 1.0 + v2 * -2.0) > theta2 else 0

    print(z)

w1 = 1.4393116261464531 
w2 = 1.205510673939901 
theta1 = 1.6901919937860768 
theta2 = 0.9288232876301696
# w1, w2, theta1, theta2 = uniform(0, 2), uniform(0, 2), uniform(0, 2), uniform(0,2)
print(w1, w2, theta1, theta2)
neuron_output(1, 1, w1, w2, theta1, theta2)
neuron_output(1, 0, w1, w2, theta1, theta2)
neuron_output(0, 1, w1, w2, theta1, theta2)
neuron_output(0, 0, w1, w2, theta1, theta2)