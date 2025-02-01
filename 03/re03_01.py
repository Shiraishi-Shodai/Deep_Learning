import numpy as np
import pandas as pd

def identity(xw):
    return xw

df = pd.read_csv("sample.tsv", sep="\t")
X = df[["x1", "x2"]].to_numpy()
w1 = np.array([[0.8, 0.9, 1.2], [0.7, -1.2, 1.1]])
w2 = np.array([-0.8, 0.2, 1.5])

u =identity( X @ w1)
z = identity( u @ w2)
print(z)
