import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('test.csv')

grad_history =[]
epoch_history =[]
eta = 0.0001
w = np.array([4.0,2.7])

for epoch in range(1,5500):
    grad =0
    for vec_x in df.to_numpy():
        x = vec_x[0]
        t = vec_x[1]
        grad += (w[0]*x+w[1]-t)*np.array([x,1.0])
   
    w = w - eta*grad
    grad_norm = np.linalg.norm(grad, ord=2)
    grad_history.append(grad_norm)
    epoch_history.append(epoch)
    print(f'{epoch} th train : w_1= {w[0]}, w_2 = {w[1]}')

    if math.sqrt(grad[0]**2 + grad[1]**2)  < 0.05:
        break

plt.xlabel("epoch")
plt.ylabel("E(loss function)")
plt.plot(epoch_history,grad_history,color="green")
plt.show()
