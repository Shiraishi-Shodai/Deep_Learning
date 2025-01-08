import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random


def fig_accuracy(th,X,y,weights,bias):
    u = np.dot(X, weights) + bias
    y_pred_sigmoid = 1 / (1 + np.exp(-u))
    y_pred = np.where(y_pred_sigmoid >= th, 1, 0)
    ac = accuracy_score(y, y_pred) 
    print(y_pred)
    return ac

def q2():
    """
    次のページのような1層のニューラルネットワークでirisデータを使い、
    sepal length = x1 ,sepal width = x2 ,
    petal length = x3 ,　petal width  = x4 　と割り当て、それぞれの重みづけ w1～w4 と、バイアスの値βは次のようにした。

    w1 = 0.44061796
    w2 = -0.90161617
    w3 = 2.31048433
    w4 = 0.9677031
    β   = -6.642248290000723

    本日配布したiris1.csvのデータを使って、この重みづけでsetosaとversicolorを分類したとき、その正解率は何％になるか求めなさい。正しい品種はlabel の列にある。
    """

    df = pd.read_csv('iris1.csv')
    print("データフレームの欠損値")
    print(df.isnull().sum())
    print("データフレームの基本情報")
    print(df.describe())

    X_train = df[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
    y_train = df["species"].map({"setosa":0,"versicolor":1}).to_numpy()

    # 特徴量の標準化(標準化するとうまく分類できないのは、おそらく先生がこの重みを求める際にX_trainを標準化しなかったから)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # print(X_train)

    # sns.pairplot(df, hue="species")
    # plt.show()
    w = np.array([0.44061796, -0.90161617, 2.31048433, 0.9677031])
    b = -6.642248290000723

    ac = fig_accuracy(0.5, X_train, y_train, w, b)
    print(f"正解率 {ac * 100} %")

def main():
    q2()


if __name__ == "__main__":
    main()