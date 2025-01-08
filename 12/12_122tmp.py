import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
# データセットを作成
np.random.seed(0)
X = np.linspace(-1,5,100) +np.random.random(100) # 2 * np.random.rand(100)
X = X.reshape(-1,1)
#y = 4 + 3 * X + np.random.randn(100)
#y = 1.5*np.sin(3.1415*X) +0.8*X - 1.5 + 0.1*np.random.randn(100)
y= X**2 -0.5 *X + 1.2+ 0.05*np.random.randn(100)
print(X.shape)
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
    e = np.mean(((model(X,w))-y)**2)
    #e = (((model(X,w))-y)**2)/len(X)
    #e =model(X,w)
    return e
# 損失関数
def loss_func2(w):
    global X,y
    e = (((model(X,w))-y)**2)/len(X)
    return e

#そのときのパラメータwと観測データXを使って予測値を出す
def predict(X,w):
    return model(X,w)

#モデルの数式
def model(X,w):
    w1=w[0]
    w2=w[1]
    y_model = X**2 -w1*X + w2  
    #y_model = w1 * X + w2
    #y_model =np.sin(3.1415*X) +w1*X +w2    
    return y_model

# 学習率
learning_rate = 0.01

# エポック数
epochs = 100

# 損失を保存するリスト
loss_history = []
w_history = []

# 勾配降下法の実装
for epoch in range(epochs):
    print("----1------")
    print(w)
    print("----2------")
    print(loss_func(w))
    print("-----3-----")
    # 勾配の計算
    grad = get_grad(loss_func,w) 
    w = w - learning_rate * grad
    w1 = w[0]
    w2 = w[1]
    # エポックごとの損失を計算
    error =predict(X,w) -y
    loss = np.mean(error ** 2)
    loss_history.append(loss)
    w_history.append([w1,w2,loss_func(np.array([w1,w2]))])
    print(f'Epoch {epoch+1}, Loss: {loss}')

# 結果の表示
print(f'\n学習されたパラメータ: w1 = {w1}, w2 = {w2}')

# 損失の推移をプロット
plt.plot(loss_history)
plt.xlabel('エポック')
plt.ylabel('損失')
plt.title('エポックごとの損失の推移（勾配降下法）')
plt.show()


w_history = np.array(w_history)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('E')
w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)
w1, w2 = np.meshgrid(w1, w2)  
E  = loss_func2([w1,w2])
print(E.shape)
ax.plot_wireframe(w1, w2, E, color='blue',linewidth=0.3)
ax.scatter(w_history[:,0], w_history[:,1], w_history[:,2], color='red')
plt.show()
