import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import japanize_matplotlib
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def sk():
    df = pd.read_csv("test17.csv")
    # print(df.head())
    X = df["X"].to_numpy().reshape(-1, 1)
    ss = StandardScaler()
    X = ss.fit_transform(X)
    t = df["t"].to_numpy()

    model = Ridge()
    model.fit(X, t)
    t_pred = model.predict(X)

    print(f"決定係数: {r2_score(t, t_pred)}") # 0.7997258594453208
    print(model.coef_, model.intercept_)

    plt.scatter(X, t)
    plt.plot(X, t_pred , c="red")
    plt.xlabel("X")
    plt.ylabel("t")
    plt.show()


def DNN():
    df = pd.read_csv("test17.csv")
    # print(df.head())
    X = df["X"].to_numpy().reshape(-1, 1)
    ss = StandardScaler()
    ss.fit_transform(X)
    T = df["t"].to_numpy()

    grad_history =[]
    epoch_history =[]
    epochs = 1000
    eta = 0.0001
    w = np.array([4.0,2.7])
    a = 0.001

    for epoch in range(1, epochs + 1):
        grad = 0
        # 勾配を計算
        for x, t in zip(X[:, 0], T):
            grad += (w[0]*x+w[1]-t)*np.array([x,1.0]) + a * np.array([w[0], w[1]])
            
        w -= eta * grad

        grad_history.append(np.linalg.norm(w, ord=2))
        epoch_history.append(epoch)
        if np.linalg.norm(w, ord=2) < 0.0001:
            break

        print(f"epoch: {epoch}, w = {w}, grad = {grad} ")

    T_pred = w[0] * X + w[1]

    # scikit-learnの結果: [30.69493941] 5.098608686919798
    print(f"決定係数: {r2_score(T, T_pred)}") # 決定係数: 0.7973333167804946
    print(f"w0 = {w[0]} w1 = {w[1]}")

    plt.title("勾配降下法による予測結果")
    plt.scatter(X, T)
    plt.plot(X, T_pred, c="red")
    plt.xlabel("X")
    plt.ylabel("T")
    plt.show()


def main():
    DNN()

if __name__ == "__main__":
    main()