import numpy as np
import random

def heavyweightFun(xw: int, theta: int):
    if xw - theta >= 0:
        return 1
    else: 
        return 0

def nuralNetWork(X, w1, w2, b1, b2):

    Xw = X @ w1
    u1 = Xw[0]
    u2 = heavyweightFun(Xw[1], b1)
    u = np.array([u1, u2])
    Xw2 = u @ w2
    z = heavyweightFun(Xw2, b2)
    return z


def main():
    w1 = np.array([[0, 1.0], [0, 1.0]])
    w2 = np.array([1.0, -2.0])
    X_list = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    correct_list = np.array([0, 1, 1, 0])
    loop = 1000

    for i in range(loop):
        w1[0, 0] = random.uniform(5, -5)
        w1[1, 0] = random.uniform(5, -5)
        b1 = random.uniform(5, -5)
        b2 = random.uniform(5, -5)
        res_list = np.array(list(map(lambda X: nuralNetWork(X, w1, w2, b1, b2), X_list)))
        if np.all(res_list == correct_list):
            print(f"計算結果: {res_list}")
            print(f"w1 = {w1}, w2 = {w2}, b1 = {b1}, b2 = {b2}の時、XORが実現できる")
            break

if __name__ == "__main__":
    main()