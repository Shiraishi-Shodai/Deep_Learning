"""
ケース3
観測点との二乗誤差を損失関数とし、単純パーセプトロンの問題として勾配降下法で解く。
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import copy

class PolynomialNeuralNetwork:
    def __init__(self, learning_rate= 0.0009, iterations=12000,verbose=True):
        self.learning_rate = learning_rate
        self.iterations = iterations #学習回数の最大値
        self.loss_curve = [] #損失関数の値をリストで保持する
        self.weights_history = []
        self.verbose = verbose

    #数値微分の関数　f はベクトルvec_wの関数である。
    def get_grad(self,h = 0.0001) :
        vec_w = np.copy(self.weights)
        f = self.loss_func()
        grad = np.zeros_like(vec_w)
        for i in range(len(vec_w)):
            vec_i_org = vec_w[i]
            vec_w[i] = vec_i_org + h
            fh1=f(vec_w)
            vec_w[i] = vec_i_org - h
            fh2=f(vec_w)
            grad[i] = (fh1-fh2)/(2*h)
            vec_w[i] = vec_i_org
        return grad

    #損失関数の無名関数を返す
    def loss_func(self):
        return lambda w : np.sum((self.y-(self.X @ w).T)**2)

    #パラメータwの更新
    def update_weights(self):
		# 勾配降下法
        grad = self.get_grad()
        self.weights = self.weights - self.learning_rate * grad
        self.weights_history.append(self.weights)
        loss_val = self.loss_func()
        self.loss_curve.append(loss_val(self.weights)) #損失を保存
    
    # 学習　全データを使う。
    def fit(self, X, y):
        # mはサンプル数、nは特徴量の次元
        self.m, self.n = X.shape
        #self.weights = np.zeros(self.n) #0で初期化
        self.weights = w = np.random.randn(self.n) #ランダムに初期化np.array([4.0,-1.2,13.0])
        self.bias = 0
        self.X = X
        self.y = y
        for epoc in range(self.iterations):
            if self.verbose :
                print(f"epoc:{epoc} , w=",end=" ")
                print(self.weights,end=" , grad = ")
                print(self.get_grad())
            
            self.update_weights()
            #if np.linalg.norm(self.weights, ord=2) <0.01:
            if math.sqrt(sum(self.weights**2))<0.01:
                break
        self.coef_=self.weights
        return True
    
    # 予測。第2引数に重みを指定するとそのときの予測値で計算する。
    def predict(self,X_test,w=[]):
        if type(w) == np.ndarray :
            pass
        else:
            w=np.array(w)
        if not len(w) :
            w = self.weights
        else:
            pass

        return X_test @ w

     
x =np.array([0.5,1,1.5,2,2.5,3,3.5,4,4.5]).reshape(-1,1)
y= np.array([6,3.5,2.1,1.5,1.9,3.6,5.8,9.6,13.6])
y_train = y
x_train = np.concatenate([x**0,x,x**2],axis=1)
#print(x_train)
#print(y_train)

model = PolynomialNeuralNetwork(verbose=False)
model.fit(x_train,y_train)
# 使用例:普通のpredictと同じように使える。
# print(model.predict(np.array([[1,0.35,0.35**2],[1,0.3,0.3**2]])))

x_t = np.linspace(-1,5,200).reshape(-1,1)
x_test = np.concatenate([x_t**0,x_t,x_t**2],axis=1)
y_pred= model.predict(x_test) #次の行でも同じ
#y_pred = model.coef_[0] + model.coef_[1]*x_t + model.coef_[2]*x_t**2 
print(model.coef_)
print(model.weights_history[50])
plt.scatter(x,y,color="green")
plt.plot(x_t,y_pred,color="red")
for i in range(0,10):
    y_pred= model.predict(x_test,model.weights_history[i*1000])
    plt.plot(x_t,y_pred,color="pink")
plt.xlabel("x")
plt.xlabel("y")
plt.title("2次式多項式での近似")
plt.show()

plt.xlabel("学習回数")
plt.ylabel("loss")
#plt.plot(epoch_history,grad_history,color="blue")
plt.plot(model.loss_curve,color="green")
plt.title("損失関数")
plt.show()