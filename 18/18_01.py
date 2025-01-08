import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df_sample = pd.read_csv("sample_2d.csv")
sample = df_sample.to_numpy()


"""
for i in range(len(sample)):
    if int(sample[i][2])==0:
        plt.scatter(sample[i][0],sample[i][1],marker="o",color="r")
    else:
        plt.scatter(sample[i][0],sample[i][1],marker="s",color="b")

        plt.scatter(sample[i][0],sample[i][1],marker="o",color="r")
"""
#cm　でカラーマップを選択する。
cm = plt.get_cmap("Spectral")
plt.scatter(sample[:,0],sample[:,1],marker="o",c=cm(sample[:,2]))
plt.show()


