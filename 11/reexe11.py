import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
import math

#数値微分の関数　f はベクトルvec_wの関数である。
def get_grad(f, vec_w, h=0.0001):
    grad = np.zeros_like(vec_w)
    for i in range(len(vec_w)):
        vec_i_org = vec_w[i]
        vec_w[i] = vec_i_org + h
        fh1 = f(vec_w)
        vec_w[i] = vec_i_org - h
        fh2 = f(vec_w)
        grad[i] = (fh1 - fh2) / (2 * h)
        vec_w[i] = vec_i_org
    return grad

def loss_fun(vec_w):
    w1 = vec_w[0]
    w2 = vec_w[1]
    w3 = vec_w[2]
    y_pred = w1 + w2 * np.sin(3.1416 * X / 5) + w3 * X
    return np.mean((y - y_pred.flatten())**2)

df = pd.read_csv("exe11.csv")
X = df['x'].to_numpy().reshape(-1,1)
y = df['y'].to_numpy()
# 標準正規分布に従う乱数を生成
rng = np.random.default_rng()
w =  rng.standard_normal(3)  
eta = 0.005
epochs = 1000
loss_history = []

for epoch in range(1, epochs + 1):
    grad = get_grad(loss_fun, w)
    w -= eta * grad
    loss = loss_fun(w) * 0.01
    loss_history.append([epoch, loss])

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, w = {w}", end=" ")
        print(f"Iteration {epoch}, Loss: {loss:.6f}")

    if np.linalg.norm(grad, ord=2) < 0.001:
        break

loss_history = np.array(loss_history)
plt.scatter(loss_history[:, 0], loss_history[:, 1], c="orange")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

y_pred = w[0] + w[1] * np.sin(3.1416 * X / 5) + w[2] * X
plt.scatter(X, y, c="blue")
plt.plot(X, y_pred, "orange")
plt.show()