import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
"""
異常検知 (Anomaly Detection)
"""
# ============================
# 1. データの準備 (MNIST)
# ============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# 28x28 => 784次元に変換
x_train = x_train.reshape((len(x_train), 28*28))
x_test  = x_test.reshape((len(x_test), 28*28))

# -- 「0~8」を正常データとし、「9」を異常データとみなす --
normal_idx_train = np.where(y_train < 9)[0]  # 0~8
normal_idx_test  = np.where(y_test < 9)[0]
anomaly_idx_test = np.where(y_test == 9)[0]  # 9

x_train_normal = x_train[normal_idx_train]
x_test_normal  = x_test[normal_idx_test]
x_test_anomaly = x_test[anomaly_idx_test]

# ============================
# 2. オートエンコーダの構築
# ============================
input_dim = 784
latent_dim = 64

encoder_input = layers.Input(shape=(input_dim,))
x = layers.Dense(256, activation='relu')(encoder_input)
encoded = layers.Dense(latent_dim, activation='relu')(x)

decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(decoder_input)
decoded = layers.Dense(input_dim, activation='sigmoid')(x)

encoder = models.Model(encoder_input, encoded, name='encoder')
decoder = models.Model(decoder_input, decoded, name='decoder')

# AE全体モデル
ae_input = layers.Input(shape=(input_dim,))
latent_vec = encoder(ae_input)
reconstructed = decoder(latent_vec)
autoencoder = models.Model(ae_input, reconstructed, name='autoencoder')

autoencoder.compile(optimizer='adam', loss='mse')

# ============================
# 3. 学習
# ============================
epochs = 10
batch_size = 256

history = autoencoder.fit(
    x_train_normal, x_train_normal,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_split=0.2
)

# ============================
# 4. 正常/異常データを再構成
# ============================
recon_normal = autoencoder.predict(x_test_normal)
recon_anomaly = autoencoder.predict(x_test_anomaly)

# 再構成誤差を計算
mse_normal = np.mean((x_test_normal - recon_normal)**2, axis=1)
mse_anomaly = np.mean((x_test_anomaly - recon_anomaly)**2, axis=1)

# ============================
# 5. 異常判定のしきい値決定
# ============================
# 例: 正常データの平均+標準偏差からしきい値を決める
threshold = np.mean(mse_normal) + 3 * np.std(mse_normal)
print("Threshold:", threshold)

# ============================
# 6. テストデータでの評価
# ============================
normal_detected   = np.sum(mse_normal < threshold)  / len(mse_normal)
anomaly_detected  = np.sum(mse_anomaly > threshold) / len(mse_anomaly)

print("Normal data correctly detected:   {:.2f}%".format(normal_detected * 100))
print("Anomaly data correctly detected:  {:.2f}%".format(anomaly_detected * 100))

# ============================
# 7. 可視化
# ============================
plt.figure(figsize=(8,4))
plt.hist(mse_normal, bins=50, alpha=0.5, label='normal')
plt.hist(mse_anomaly, bins=50, alpha=0.5, label='anomaly')
plt.axvline(threshold, color='red', linestyle='--', label='threshold')
plt.title("Reconstruction Error Distribution")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend()
plt.show()
