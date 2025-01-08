import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
# データセットを作成
np.random.seed(0)
X = 2 * np.random.rand(100)
y = 4 + 3 * X + np.random.randn(100)

# パラメータの初期化
w1 = np.random.randn()
w2 = np.random.randn()
w =np.array([w1,w2])

#数値微分の関数　f はベクトルvec_wの関数である。
def get_grad(f,vec_w,h=0.0001) :
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

# 損失関数
def loss_func(w):
    global X,y
    w1=w[0]
    w2=w[1]
    e = np.mean(((w1 * X + w2)-y)**2)
    #print(e)
    return e

#そのときのパラメータwと観測データXを使って予測値を出す
def predict(X,w):
    w1=w[0]
    w2=w[1] 
    return w1 * X + w2

# 学習率
learning_rate = 0.01

# エポック数
epochs = 100

# 損失を保存するリスト
loss_history = []

# 勾配降下法の実装
for epoch in range(epochs):
    # 勾配の計算
    grad = get_grad(loss_func,w) 
    w = w - learning_rate * grad
    w1 = w[0]
    w2 = w[1]
    # エポックごとの損失を計算
    error =predict(X,w) -y
    loss = np.mean(error ** 2)
    loss_history.append(loss)
    print(f'Epoch {epoch+1}, Loss: {loss}')

# 結果の表示
print(f'\n学習されたパラメータ: w1 = {w1}, w2 = {w2}')

# 損失の推移をプロット
plt.plot(loss_history)
plt.xlabel('エポック')
plt.ylabel('損失')
plt.title('エポックごとの損失の推移（勾配降下法）')
plt.show()

# データとモデルのプロット
plt.scatter(X, y, color='blue', label='データ')
plt.plot(X,predict(X,w), color='red', label='学習されたモデル')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('データと学習されたモデルの比較（勾配降下法）')
plt.show()
