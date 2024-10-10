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

def q3():

    """
    次のページのような1層のニューラルネットワークでiris2.csvのデータを使い、sepal length = x1 ,sepal width = x2 ,
    petal length = x3 ,　petal width  = x4 　と割り当て、virginica とversicolor を分類するモデルを作った。
    w1～w3 の値をひとまず、下記のようにした。

    w1 =  0.39443136
    w2 =  0.51327025
    w3 =  -2.93075043

    virginica とversicolor の分類の正解率が96％以上になるような
    w4 とβの組を１つ見つけなさい。
    """
    
    df = pd.read_csv('iris2.csv')
    print("データフレームの欠損値")
    print(df.isnull().sum())
    print("データフレームの基本情報")
    print(df.describe())

    # sns.pairplot(df, hue="species")
    # plt.show()

    X_train = df[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
    y_train = df["species"].map({"virginica":0,"versicolor":1}).to_numpy()

    w123 = np.array([0.39443136, 0.51327025, -2.93075043])
    w4 =0
    b = 0

    AC = 0.96 # 目標正解率
    current_ac = 0

    while current_ac < AC:
        w4 = random.uniform(20, -20)
        w = np.append(w123, w4)
        b = random.uniform(20, -20)

        current_ac = fig_accuracy(0.5, X_train, y_train, w, b)
        print(current_ac)
        
    print(f"発見したw4 = {w4}")
    print(f"発見したβ = {b}")


def main():
    q3()


if __name__ == "__main__":
    main()