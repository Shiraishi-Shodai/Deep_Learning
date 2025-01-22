import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

"""
MNIST (手書き数字データセット)を用いて、
シンプルな全結合(MLP)型のオートエンコーダを実装する。
"""
# -----------------------------
# 1. データの準備
# -----------------------------
# MNISTデータセットをダウンロード・読み込み
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# ピクセル値を [0,1] に正規化し、(batch, 784) の形状に変換
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# (batch_size, 28, 28) -> (batch_size, 784)
x_train = x_train.reshape((len(x_train), 28 * 28))
x_test  = x_test.reshape((len(x_test), 28 * 28))

# -----------------------------
# 2. モデルの定義 (Encoder/Decoder)
# -----------------------------

# 入力次元(784) → 潜在次元(64) のエンコーダ
latent_dim = 64

encoder_input = layers.Input(shape=(784,))
# Encoder ネットワーク：784 -> 256 -> 64
encoded = layers.Dense(256, activation='relu')(encoder_input)
encoded = layers.Dense(latent_dim, activation='relu')(encoded)

# Decoder ネットワーク：64 -> 256 -> 784
decoder_input = layers.Input(shape=(latent_dim,))
decoded = layers.Dense(256, activation='relu')(decoder_input)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# エンコーダとデコーダをそれぞれモデル化
encoder = models.Model(encoder_input, encoded, name='encoder')
decoder = models.Model(decoder_input, decoded, name='decoder')

# 最終的なオートエンコーダモデル (encoder -> decoder)
autoencoder_input = layers.Input(shape=(784,))
encoded_output = encoder(autoencoder_input)
decoded_output = decoder(encoded_output)
autoencoder = models.Model(autoencoder_input, decoded_output, name='autoencoder')

# -----------------------------
# 3. 学習フェーズ
# -----------------------------
# 損失関数は MSE (再構成誤差) または binary_crossentropy を使用可能
autoencoder.compile(optimizer='adam', loss='mse')

epochs = 10
batch_size = 256

history = autoencoder.fit(
    x_train, x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

# -----------------------------
# 4. 再構成結果の可視化
# -----------------------------
# テストデータのうちから任意の画像を10枚再構成してみる
decoded_imgs = autoencoder.predict(x_test[:10])

# 可視化用に画像を (28,28) に戻す
x_test_imgs = x_test[:10].reshape(-1, 28, 28)
decoded_imgs = decoded_imgs.reshape(-1, 28, 28)

n = 10  # 表示枚数
plt.figure(figsize=(20, 4))
for i in range(n):
    # 入力画像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_imgs[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # 再構成画像
    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(decoded_imgs[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()