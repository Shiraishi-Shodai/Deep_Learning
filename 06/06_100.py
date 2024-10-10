import numpy as np

# ロジスティック回帰で2値（0/1）分類するための自作クラス
class MyLogisticRegression:
	#コンストラクタ。学習係数と繰り返し回数を指定する。
	def __init__(self, learning_rate=0.01, iterations=1000):
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.loss_curve = [] #損失関数の値をリストで保持する
	
	# シグモイド関数（というかロジスティック関数）
	def sigmoid(self, z ,k=1 ):
		return 1 / (1 + np.exp(-k*z))
	
	# 損失関数 -Σ(y_n log(yhat_n)+(1-y_n)*log(1-yaht_n))
	def loss_func(self,y_pred,y):
		E =-np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
		return E
	
	#重みの更新
	def update_weights(self):
		linear_model = np.dot(self.X, self.weights) + self.bias
		y_predicted = self.sigmoid(linear_model)

		#クロスエントロピー誤差を偏微分するとわかるが、 ∂E/∂w = x * (y_hat - y)
		dw = np.dot(self.X.T, (y_predicted - self.y)) / self.m
		db = np.sum(y_predicted - self.y)/ self.m

		self.loss_curve.append(self.loss_func(y_predicted,self.y)) #損失を保存
		# 勾配降下法
		self.weights = self.weights - self.learning_rate * dw
		self.bias = self.bias - self.learning_rate * db
	
	# 学習　全データを使う。
	def fit(self, X, y):
		self.m, self.n = X.shape
		# mはサンプル数、nは特徴量の次元
		#self.weights = np.zeros(self.n) #0で初期化 まずはこの行を使い、下の行をコメントアウトしてみよ。
		self.weights = np.random.randn(self.n) #ランダムに初期化
		self.bias = 0
		self.X = X
		self.y = y
		for _ in range(self.iterations):
			self.update_weights()

	# 予測
	def predict(self, X):
		#linear_model = np.dot(X, self.weights) + self.bias 以下はこれと同じ　@を使っている
		linear_model = X @ self.weights + self.bias
		y_predicted = self.sigmoid(linear_model)
		return np.array([1 if i > 0.5 else 0 for i in y_predicted])

# 使用例
if __name__ == "__main__":
	from sklearn.metrics import accuracy_score
	import pandas as pd 
	#ケース2 virginicaとversicolorを分類する
	iris_data = pd.read_csv("iris2.csv")
	X = iris_data[["sepal_length","sepal_width","petal_length","petal_width"]].to_numpy()
	y = iris_data["species"].map({"virginica":0,"versicolor":1}).to_numpy()
	
	model = MyLogisticRegression(learning_rate=0.05,iterations=800)
	#model = LogisticRegression(learning_rate=0.01) # learning_rate=0.01では94％くらい
	model.fit(X, y)
	print(f"weights = {model.weights}")
	print(f"bias = {model.bias}")
	print(f"----------------------------")

	# 予測
	y_pred = model.predict(X)
	print(f"true label = {y}")
	print(f"prediction = {y_pred}")
	print(f"----------------------------")
	print(f"正解率={accuracy_score(y_pred,y)}%")
	import matplotlib.pyplot as plt
	import japanize_matplotlib
	plt.plot(model.loss_curve)
	plt.title("損失関数")
	plt.xlabel("学習回")
	plt.ylabel("loss")
	plt.show()
"""
weights = [ 1.86325507  2.0263241  -3.13379069 -1.85961275]
bias = 0.9126093463916441
----------------------------
true label = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
prediction = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
----------------------------
正解率=0.98%
"""