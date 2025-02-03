import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv("iris2.csv")
X = iris_data[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
y = iris_data["species"].map({"virginica":0,"versicolor":1}).to_numpy()

#th :閾値
def fig_accuracy(th,x,y,weights,bias):
    # 線形結合を計算 (weights * X_test + bias)
    linear_output = np.dot(X, weights) + bias

    # シグモイド関数を適用して確率を計算
    y_pred_sigmoid = 1 / (1 + np.exp(-linear_output))
    y_pred = list(map(lambda x: 1 if x> th else 0, y_pred_sigmoid))
    #print(y_pred_sigmoid)
    #print(y_pred)
    #print(y_proba)
    #print(np.concatenate([y_pred_sigmoid.reshape(-1,1),y.reshape(-1,1)],axis=1))

    # 一致率
    accuracy = accuracy_score(y, y_pred)
    return accuracy

for w4 in np.arange(0.01, 0.95, 0.01):
    for bias in np.arange(10, 15, 0.01):
        weights =np.array([ 0.39443136 , 0.51327025,-2.93075043 ,w4 ])
        acc = fig_accuracy(0.5,X,y,weights,bias)
        if acc > 0.94:
            print(f"w4={w4},bias={bias} => Accuracy:{acc}")
            break

"""
[ 0.39443136  0.51327025 -2.93075043 -2.4170433 ]
14.430804325583564
"""