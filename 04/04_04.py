import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import pandas as pd

df = pd.read_csv("sample.tsv",sep="\t")
x1 = df["x1"].to_numpy().reshape(-1,1)
x2 = df["x2"].to_numpy().reshape(-1,1)
X = np.concatenate([x1, x2], axis =1)
w11 = 0.8
w12 = 0.9
w13 = 1.2
w21 = 0.7
w22 = -1.2
w23 = 1.1
w2_1 = -0.8
w2_2 = 0.2
w2_3 = 1.5

#中間のノードのバイアス
b_11 = 0
b_12 = 0
b_12 = 0

#出力層のノードのバイアス
b3 =0 
#入力層から中間層への重み
W_1 = np.array([[w11, w12, w13], [w21, w22, w23]])
b_1 = np.array([b_11, b_12, b_12])
b_2 =np.array([b3])
W_2 = np.array([[w2_1], [w2_2], [w2_3]])
u_in = X@W_1 + b_1

u_out=u_in #ここで中間層の活性化関数を適用する

print(u_in)
print(u_out@W_2+b_2)

z = u_out #本来であればここで出力層の活性化関数を適用する