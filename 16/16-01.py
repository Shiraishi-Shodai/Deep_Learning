import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
df = pd.read_csv('test.csv')

plt.scatter(df["X"],df["t"])
plt.xlabel("X")
plt.xlabel("t")
plt.title("Xとtの関係")
plt.show()