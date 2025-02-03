import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv("iris2.csv")
X = iris_data[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
y = iris_data["species"].map({"virginica":0,"versicolor":1}).to_numpy()

weights = np.array([ 0.39443136, 0.51327025 , -2.93075043 , 0 ])
bias = 0
th = 0.5
eligibility = 0.94

def calc_acc(X, w, b, th, y):
    linear_output = np.dot(X, w) + b
    y_pred_sigmoid = 1 / (1 + np.exp(-linear_output))
    y_pred = list(map(lambda x: 1 if x > th else 0, y_pred_sigmoid))
    acc = accuracy_score(y, y_pred)
    return acc

for w4 in np.arange(0.1, 0.95, 0.01):
    for b in np.arange(1, 20, 0.01):
        weights[3] = w4
        bias = b
        acc = calc_acc(X, weights, bias, th, y)
        print(f"正解率 : {acc * 100}%")
        if acc > eligibility:
            print(f"wight: {weights}, bias: {bias}")
            break
    else:
        print("breakせずに内側のfor文を終了")
        continue
    print("breakしてfor文を終了")
    break