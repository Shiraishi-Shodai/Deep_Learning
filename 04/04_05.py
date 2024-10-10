import numpy as np
from sympy import *

x1, x2, w11, w12, w13, w21, w22, w23 = symbols('x1, x2, w11, w12, w13, w21, w22, w23')
b1, b2, b3 = symbols('b1, b2, b3')
z1, z2, z3 = symbols('z1, z2, z3')
w2_1, w2_2, w2_3 = symbols('w2_1, w2_2, w2_3')

X = np.array([[x1,x2],[10,20],[100,200]])
W = np.array([[w11, w12, w13], [w21, w22, w23]])
b = np.array([b1, b2, b3])
Z = X@W+b
print(Z)
W_2 = np.array([[w2_1], [w2_2], [w2_3]])
Z = np.array([[z1,z2,z3],[1,2,3],[10,20,30]])
b_2 =np.array([b3])
print(Z@W_2+b_2)

# Z = X@W+b
# print(Z@W_2+b_2)
