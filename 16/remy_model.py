import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import japanize_matplotlib

# 損失関数(平均二乗誤差) Xは(101, 1)
def mse( X, t, w):
    # tを2次元にして計算させる。理由はa.pyまで
    t_pred = w[0] * X + w[1]
    return np.mean((t_pred - t.reshape(-1, 1))**2)

# 損失関数を微分したもの Xは(101, 1)
def mse_derivative(X, t, w):
    n = len(X)
    y_pred = X * w[0] + w[1]
    tail = np.concatenate([X, np.ones((n, 1))], axis=1) # 末尾でw1とw2にそれぞれかける値
    grad = (2 / n) * np.sum((y_pred - t.reshape(-1, 1)) * tail, axis=0)  
    return grad

class SimplePerseptron():
    def __init__(self, eta=0.001, iterations=1000):
        self.eta = eta
        self.iterations = iterations
        self.loss_history = []
    
    def foward(self):
        loss = mse(self.X, self.t, self.w)
        return loss

    def backward(self):
        grad = mse_derivative(self.X, self.t, self.w)
        return grad

    def fit(self, X, t, weight_size):
        # 標準正規分布にしたがう重みを初期化
        rng = np.random.default_rng()
        self.w = np.array([4.0,2.7]) #rng.standard_normal(weight_size)
        self.X = X
        self.t = t

        for epoch in range(1, self.iterations + 1):
            loss = self.foward()
            grad = self.backward()
            self.w -= self.eta * grad

            self.loss_history.append([epoch, loss])
            if epoch % 100  == 0:
                print(f"Epoch {epoch} loss = {loss}")
            
            # if np.linalg.norm(self.w, ord=2) < 0.001:
            #     break
        self.loss_history = np.array(self.loss_history)
        plt.title("SimplePerseptronクラスによる学習の推移")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(self.loss_history[:, 0], self.loss_history[:,1], c="orange")
        plt.savefig("simple_perseptron.png")
        plt.show()

    def preditc(self, testX, testT):
        t_pred = self.w[0] * testX + self.w[1]
        t_pred = t_pred.flatten() # グラフに描画する前に1次元科

        plt.title("SimplePerseptronクラスによる予測")
        plt.xlabel("X")
        plt.ylabel("t")
        plt.scatter(testX, testT, c="blue")
        plt.plot(testX, t_pred, c="orange")
        plt.savefig("simple_perseptron2.png")
        plt.show()
        return t_pred

def main():
    df = pd.read_csv('test.csv')
    X = df["X"].to_numpy().reshape(-1, 1)
    t = df["t"].to_numpy()
    testX = X
    testT = t

    model = SimplePerseptron(eta=0.001, iterations=5000)
    weight_size = 2
    model.fit(X, t, weight_size)

    model.preditc(testX, testT)

if __name__ == "__main__":
    main()