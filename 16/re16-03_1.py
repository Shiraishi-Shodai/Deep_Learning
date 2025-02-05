import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('test.csv')

grad_history =[]
epoch_history =[]
epochs = 5000
eta = 0.001
w = np.array([4.0,2.7])
# データの行数
n = len(df.to_numpy())
# 真のデータ
t = df["t"].to_numpy()
# 入力データ
X = df["X"].to_numpy()

for epoch in range(1, epochs + 1):

    y_pred = np.array([w[0] * xi + w[1] for xi in X])
    tail = np.concatenate([X.reshape(-1, 1), np.ones_like(X).reshape(-1, 1)], axis=1) # 末尾でw1とw2にそれぞれかける値

    grad = (2 / n) * np.sum((y_pred - t).reshape(-1, 1) * tail, axis=0)  
    w -= eta * grad
    grad_norm = np.linalg.norm(grad, ord=2)
    grad_history.append(grad_norm)
    epoch_history.append(epoch)

    error = np.mean((w[0] * X + w[1] - t)**2)
    print(f"Error : {error}")
    print(f'{epoch} th train : w_1= {w[0]}, w_2 = {w[1]}')

    if math.sqrt(grad[0]**2 + grad[1]**2)  < 0.05:
        break

plt.xlabel("epoch")
plt.ylabel("E(loss function)")
plt.plot(epoch_history,grad_history,color="green")
plt.show()