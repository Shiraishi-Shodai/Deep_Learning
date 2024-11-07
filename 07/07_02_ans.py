import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

iris_data = pd.read_csv("iris1.csv")
X = iris_data[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
y = iris_data["species"].map({"setosa":0,"versicolor":1}).to_numpy()

weights = np.array([ 0.44061796, -0.90161617 , 2.31048433 , 0.9677031 ])
bias = -6.642248290000723

# 線形結合を計算 (weights * X_test + bias)
linear_output = np.dot(X, weights) + bias

# シグモイド関数を適用して確率を計算
y_pred_sigmoid = 1 / (1 + np.exp(-linear_output))
y_pred = list(map(lambda x: 1 if x> 0.5 else 0, y_pred_sigmoid))
#print(y_pred_sigmoid)
#print(y_pred)
#print(y_proba)
#print(np.concatenate([y_pred_sigmoid.reshape(-1,1),y.reshape(-1,1)],axis=1))

# 一致率
accuracy = accuracy_score(y, y_pred)
print("Accuracy:",accuracy)