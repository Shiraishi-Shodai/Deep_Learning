import numpy as np
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


