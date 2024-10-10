from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

#ケース1 setosa とversicolorを分類する
iris_data = pd.read_csv("iris1.csv")
X = iris_data[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
y = iris_data["species"].map({"setosa":0,"versicolor":1}).to_numpy()

#ケース2 virginicaとversicolorを分類する
iris_data = pd.read_csv("iris2.csv")
X = iris_data[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
y = iris_data["species"].map({"virginica":0,"versicolor":1}).to_numpy()

#今回は、全データを訓練データとして使う
X_train=X
y_train=y

# 特徴量の標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# ロジスティック回帰モデルの定義と訓練
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 作ったモデルで訓練データの予測分類をさせる
y_pred = model.predict(X_train)

# 正答率の計算
accuracy = accuracy_score(y_train, y_pred)
print("Accuracy:",accuracy)

# 重み（coef_）とバイアス（intercept_）を取得
weights = model.coef_#[0]
bias = model.intercept_#[0]

print("Weights (重み):", weights)
print("Bias (バイアス):", bias)

y_proba = model.predict_proba(X_train)

# 線形結合を計算 (weights * X_test + bias)
linear_output = np.dot(X_train, weights[0]) + bias[0]
# シグモイド関数を適用して確率を計算
y_hat_sigmoid = 1 / (1 + np.exp(-linear_output))
#print(y_hat_sigmoid)
#print(y_pred)
#print(y_proba)

import matplotlib.pyplot as plt
#print(np.concatenate([y_hat_sigmoid.reshape(-1,1),y_pred.reshape(-1,1),y_proba],axis=1))
# plt.plot(model.loss_curve_)
# plt.xlabel('Iteration')
# plt.ylabel('loss')
# plt.grid(True)
# plt.show()

print(y_train)