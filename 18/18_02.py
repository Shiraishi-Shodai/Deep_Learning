import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

# データ読み込み
df_sample = pd.read_csv("sample_2d_02.csv")
#df_sample = pd.read_csv("sample_2d.csv")
sample = df_sample.to_numpy()

def plot_boundary(model, X, Y, target, xlabel, ylabel):
    cmap_dots = ListedColormap([ "#1f77b4", "#ff7f0e", "#2ca02c"])
    cmap_fills = ListedColormap([ "#c6dcec", "#ffdec2", "#cae7ca"])
    #ステップ関数
    def step_func(x,theta=0.5):
        if x >= theta :
            return 1.0
        else :
            return 0.0

    plt.figure(figsize=(5, 5))
    vfunc = np.vectorize(step_func)
    if 1:
        XX, YY = np.meshgrid(
            np.linspace(X.min()-1, X.max()+1, 200),
            np.linspace(Y.min()-1, Y.max()+1, 200))
        #pred = vfunc(vec_w[0]*XX + vec_w[1]*YY+vec_w[2])
        pred = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        plt.pcolormesh(XX, YY, pred, cmap=cmap_fills, shading="auto")
        plt.contour(XX, YY, pred, colors="gray") 
    plt.scatter(X, Y, c=target, cmap=cmap_dots)
    #print("-------------------------------")
    #print(pred)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# パラメータ設定
num_epochs = 1 
num_epochs = 5
num_epochs = 10
#num_epochs = 20
num_epochs = 50
num_epochs = 100

# モデル作成
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
 
print(model.summary())
# トレーニング（分類）
data = sample[:,0:2]
labels = sample[:,2].reshape(-1, 1)
model.fit(data, labels, epochs=num_epochs, batch_size=10)
 
# 分類結果出力
#predicted_classes = model.predict_classes(data, batch_size=10)
predicted_classes = model.predict(data, batch_size=10)
#print(predicted_classes)
# 分類結果可視化
for i in range(len(sample)):
    # 分類結果を色で表示
    if predicted_classes[i] < 0.51:
        target_color = "r"
    else:
        target_color = "b"
    # 実際のクラスをマーカーで表示
    if int(sample[i][2])==0:
        target_marker = "o"
    else:
        target_marker = "s"
    plt.scatter(sample[i][0],sample[i][1],marker=target_marker,color=target_color)
plt.show()

plot_boundary(model,data[:,0],data[:,1],labels,"X","Y")