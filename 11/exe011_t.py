import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

df = pd.read_csv("exe11.csv")
X = df['x'].to_numpy().reshape(-1, 1)
y = df['y'].to_numpy()

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

# 損失関数: wの関数
def loss_func(w):
    global X,y  #お行儀が悪いけどグローバル変数
    y_pred = w[0] + w[1] * np.sin(3.1416 * X / 5) + w[2] * X
    loss = np.mean((y - y_pred.flatten()) ** 2)
    return loss

# 最適な学習率を求める
learning_rates = np.linspace(0.001, 0.01, 10)
min_loss = float('inf')
best_learning_rate = None
best_w = None

# 学習率ごとに評価
for lr in learning_rates:
    w = np.array([1.0, 1.0, 1.0]) #初期値
    max_iterations = 1000

    # 勾配降下法
    for i in range(1, max_iterations + 1):
        grad = get_grad(loss_func, w)
        w -= lr * grad

        # 勾配降下法を止める条件
        if np.linalg.norm(grad) < 1e-6:
            break

    # 損失を計算
    current_loss = loss_func(w)
    if current_loss < min_loss:
        min_loss = current_loss
        best_learning_rate = lr
        best_w = w

# 最適な学習率とパラメータを出力
print(f"最適な学習率: {best_learning_rate:.3f}")
a1, a2, a3 = best_w
print(f"最適なパラメータ: a1 = {a1:.5f}, a2 = {a2:.5f}, a3 = {a3:.5f}")

y_pred = a1 + a2 * np.sin(3.1416 * X / 5) + a3 * X

plt.plot(X, y_pred, color="red", label="フィッティング曲線")  # 予測曲線
plt.scatter(X, y, color="blue", label="元のデータ")  # CSVから与えられたデータ
plt.show()
