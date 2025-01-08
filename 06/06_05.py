import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#ケース1 setosa とversicolorを分類する
iris_data = pd.read_csv("iris_original.csv")
X = iris_data[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
y = iris_data["species"]#.map({"setosa":0,"versicolor":1,"virginica":2}).to_numpy()

sns.pairplot(iris_data, hue="species")
plt.show()