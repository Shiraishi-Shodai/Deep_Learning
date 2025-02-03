import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import japanize_matplotlib
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris1.csv")

X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].map({"setosa": 0, "versicolor": 1}).to_numpy()
w = np.array([0.44061796, -0.90161617, 2.31048433, 0.9677031])
b   = np.array([-6.642248290000723])

# ニューロンに入ってきた重みと入力値の線型結合を計算
linear_output = np.dot(X, w) + b
# plt.title("linear_outputの分布")
# plt.plot(np.linspace(linear_output.min(), linear_output.max(), len(linear_output)), linear_output)
# plt.show()

# 線形結合した結果をシグモイド関数で0 ~ 1の範囲に圧縮して分類にしようする確率を計算
y_pred_sigmoid = 1 / (1 + np.exp(-linear_output))
# print(y_pred_sigmoid)
# plt.title("y_pred_sigmoidの分布")
# plt.plot(np.linspace(linear_output.min(), linear_output.max(), len(linear_output)), y_pred_sigmoid)
# plt.show()

y_pred = np.array(list(map(lambda x: 1 if x > 0.5 else 0, y_pred_sigmoid)))
print(f"正解率: {accuracy_score(y, y_pred) * 100}%")
