import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import japanize_matplotlib

def nizishiki(X, y):
    x1 = X.reshape(-1, 1)
    x2 = (X**2).reshape(-1, 1)

    X_train = np.concatenate([x1, x2], axis=1)
    y_train = y

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    w1 = model.intercept_
    w2, w3 = model.coef_[0], model.coef_[1]

    print("2次式としたとき")
    print(f"w1: {w1}, w2: {w2}, w3: {w3}")
    print(f"決定係数: {r2_score(y, y_pred) * 100} %", end="\n\n")

    plt.title("yはXの2次式としたとき")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.plot(X, y_pred, c="r", label="予測値")
    plt.scatter(X, y, label="実測値")
    plt.legend()
    plt.show()



def sanzishiki(X, y):
    x1 = X.reshape(-1, 1)
    x2 = (X**2).reshape(-1, 1)
    x3 = (X**3).reshape(-1, 1)


    X_train = np.concatenate([x1, x2, x3], axis=1)
    y_train = y

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    w1 = model.intercept_
    w2, w3, w4 = model.coef_[0], model.coef_[1], model.coef_[2]

    plt.title("yはXの3次式としたとき")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.plot(X, y_pred, c="r", label="予測値")
    plt.scatter(X, y, label="実測値")
    plt.legend()
    plt.show()

    print("3次式としたとき")
    print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}")
    print(f"決定係数: {r2_score(y, y_pred) * 100} %")

def main():
    X = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    y = np.array([6, 3.5, 2.1, 1.5, 1.9, 3.6, 5.8, 9.6, 13.6])

    nizishiki(X, y)
    sanzishiki(X, y)


if __name__ == "__main__":
    main()


