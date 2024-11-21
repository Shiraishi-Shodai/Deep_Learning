import tensorflow as tf
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

"""
ValueError: This model has not yet been built. Build the model first by calling build() or by calling the model on a batch of data.
"""
# model.summary() を呼び出す時点で、Sequential モデルには入力形状が明示されていないためです。
# Dense や Flatten 層は入力形状を指定しない場合、データがモデルに一度渡されるまで自動的にビルドされません。
# このコードでは、モデルがビルドされるのは model.fit() の呼び出し時です。
# しかし、その前に model.summary() を呼び出しているため、エラーが発生しています。

start = time.time()
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 先生の古いコード
"""
model = tf.keras.models.Sequential([
  # tf.keras.layers.Flatten(), # 画像を1次元にする
   tf.keras.layers.Flatten(input_shape=(28, 28)),  # 入力形状を指定
  tf.keras.layers.Dense(512, activation=tf.nn.relu), # 中間層512個のノード。活性化関数はrelu
  tf.keras.layers.Dropout(0.2), # 前の層から来た情報を捨てる。過学習を防ぐため(0 から 1 の間の浮動小数点数。ドロップする入力単位の端数。)
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # one hot encodingで分類するから10クラス。中間層10個のノード。活性化関数はsoftmax
])
"""

# 最新の公式コード
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# CategoricalCrossentropy ではラベルが ワンホット形式（例: [1, 0, 0, ..., 0]）であることを期待しています。
# loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test,verbose=2)

p_time = time.time() - start
print(f'処理時刻={p_time}秒')