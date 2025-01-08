import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("test17.csv")
# print(df.head())
X = df["X"].to_numpy().reshape(-1, 1)
ss = StandardScaler()
X = ss.fit_transform(X)
t = df["t"].to_numpy()

model = Ridge()
model.fit(X, t)
t_pred = model.predict(X)

print(f"決定係数: {r2_score(t, t_pred)}") # 0.7997258594453208
print(model.coef_, model.intercept_)

plt.scatter(X, t)
plt.plot(X, t_pred , c="red")
plt.xlabel("X")
plt.ylabel("t")
plt.show()
