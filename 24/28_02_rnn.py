import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam

#シード値の固定
seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

#データの読み込み
data = np.loadtxt("./sin_data.csv")
plt.plot(data[:500])
plt.show()

#history_stepsステップ数を入力に使い、future_stepsステップ数を予測する様にデータを加工する
def create_dataset(data, history_steps, future_steps):
    input_data = []
    output_data= []
    
    for i in range(len(data)-history_steps-future_steps):
        input_data.append([[val] for val in data[i:i+history_steps]])
        output_data.append(data[i+history_steps:i+history_steps+future_steps])
    
    return np.array(input_data), np.array(output_data)

train_data = data[:int(len(data) * 0.75)]
test_data = data[int(len(data) * 0.75):]

#10ステップ分のデータから５ステップ未来までを予測するようなデータを作成する
history_steps = 10
future_steps = 5
x_train, y_train = create_dataset(train_data, history_steps, future_steps)
x_test, y_test   = create_dataset(test_data, history_steps, future_steps)  

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



#モデルの構築
model_rnn = Sequential()  
model_rnn.add(SimpleRNN(units=future_steps, input_shape=(history_steps,1),return_sequences=False))
model_rnn.add(Dense(future_steps,activation="linear"))  
model_rnn.compile(optimizer = Adam(lr=0.001), loss="mean_squared_error",)

#モデルの構造を表示する
print(model_rnn.summary())

#学習開始
history = model_rnn.fit(x_train, y_train, batch_size=32, epochs=500, verbose=1)

#学習したモデルで予測をする
y_pred = model_rnn.predict(x_test)

plt.figure(figsize=(15,5))
#青色で予測値、オレンジ色で実際の値を表示
plt.plot([p[0] for p in y_pred],color="blue",label="pred")
plt.plot([p[0] for p in y_test],color="orange",label="actual")
plt.legend()
plt.show()