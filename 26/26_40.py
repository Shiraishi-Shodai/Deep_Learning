import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

# ==========================================================
# 1. データ (今回は先ほどと同様に MNIST を使用)
# ==========================================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

x_train = x_train.reshape((len(x_train), 28*28))
x_test  = x_test.reshape((len(x_test), 28*28))

normal_idx_train = np.where(y_train < 9)[0]  # 0~8 -> 正常
normal_idx_test  = np.where(y_test < 9)[0]
anomaly_idx_test = np.where(y_test == 9)[0]  # 9 -> 異常

x_train_normal = x_train[normal_idx_train]
x_test_normal  = x_test[normal_idx_test]
x_test_anomaly = x_test[anomaly_idx_test]

# ==========================================================
# 2. Sampling Layer (Reparameterization Trick)
# ==========================================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ==========================================================
# 3. Encoder (推論モデル)
# ==========================================================
latent_dim = 16

encoder_inputs = layers.Input(shape=(28*28,))
x = layers.Dense(128, activation='relu')(encoder_inputs)
x = layers.Dense(64, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
z = Sampling()([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# ==========================================================
# 4. Decoder (生成モデル)
# ==========================================================
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(64, activation='relu')(latent_inputs)
x = layers.Dense(128, activation='relu')(x)
decoder_outputs = layers.Dense(28*28, activation='sigmoid')(x)
decoder = Model(latent_inputs, decoder_outputs, name='decoder')

# ==========================================================
# 5. VAE クラス定義
# ==========================================================
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstructed = self.decoder(z)

            # 再構成誤差
            #recon_loss = tf.keras.losses.mean_squared_error(data, reconstructed)
            recon_loss = tf.keras.losses.MSE(data, reconstructed) # Changed to MSE
            #recon_loss = tf.reduce_mean(tf.reduce_sum(recon_loss, axis=1))
            recon_loss = tf.reduce_mean(recon_loss) # Calculating the mean across all samples in the batch

            # KLダイバージェンス
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1
                )
            )

            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer='adam', loss=lambda x, y: x) # Adding a dummy loss function


# 学習 (正常データのみ)
vae.fit(
    x_train_normal, x_train_normal,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.2
)

# ==========================================================
# 6. 異常検知 (再構成誤差の比較)
# ==========================================================
# 正常データを再構成
recon_normal = vae.predict(x_test_normal)
mse_normal = np.mean((x_test_normal - recon_normal)**2, axis=1)

# 異常データを再構成
recon_anomaly = vae.predict(x_test_anomaly)
mse_anomaly = np.mean((x_test_anomaly - recon_anomaly)**2, axis=1)

# しきい値決定 (簡易版)
threshold = np.mean(mse_normal) + 3 * np.std(mse_normal)

print("Threshold:", threshold)

normal_detected  = np.sum(mse_normal < threshold) / len(mse_normal)
anomaly_detected = np.sum(mse_anomaly > threshold) / len(mse_anomaly)

print("Normal data correctly detected:  {:.2f}%".format(normal_detected * 100))
print("Anomaly data correctly detected: {:.2f}%".format(anomaly_detected * 100))

# ==========================================================
# 7. 結果の可視化
# ==========================================================
plt.figure(figsize=(8,4))
plt.hist(mse_normal, bins=50, alpha=0.5, label='normal')
plt.hist(mse_anomaly, bins=50, alpha=0.5, label='anomaly')
plt.axvline(threshold, color='red', linestyle='--', label='threshold')
plt.title("Reconstruction Error Distribution (VAE)")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend()
plt.show()
