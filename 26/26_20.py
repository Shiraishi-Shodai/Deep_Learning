import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

# =====================================================================
# 1. MNISTデータセットの準備
# =====================================================================
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# (28,28) -> (28,28,1) の形に拡張
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)

# =====================================================================
# 2. Sampling Layer (再パラメータ化トリック)
# =====================================================================
class Sampling(layers.Layer):
    """
    潜在空間の平均 (z_mean) と対数分散 (z_log_var) から
    z = z_mean + exp(0.5 * z_log_var) * eps
    をサンプリングするためのカスタムLayer。
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        # epsilon ~ N(0, I)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# =====================================================================
# 3. Encoder (推論モデル) の定義
# =====================================================================
latent_dim = 2  # 潜在次元を2にして可視化を容易にする例

encoder_inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)

# z_meanとz_log_varを出力
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# 再パラメータ化
z = Sampling()([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# =====================================================================
# 4. Decoder (生成モデル) の定義
# =====================================================================
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

decoder = Model(latent_inputs, decoder_outputs, name='decoder')
decoder.summary()

# =====================================================================
# 5. VAE Model の定義 (Encoder + Decoder)
# =====================================================================
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        """
        推論時は単純に encoder -> decoder の順序で呼び出し、
        再構成画像 (decoderの出力) を返す。
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        """
        Kerasの学習ステップをオーバーライドし、
        再構成誤差 + KLダイバージェンスを計算する。
        """
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # 再構成誤差 (ここでは binary_crossentropy を使用)
            # pixel-wiseの平均をとり、最後にバッチ方向の平均も取る
            # Kerasのbinary_crossentropyは出力の形状に注意
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )

            # KLダイバージェンス (KL(p||q))
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

# VAEインスタンスを生成
vae = VAE(encoder, decoder)

# =====================================================================
# 6. コンパイルと学習
# =====================================================================
vae.compile(optimizer=tf.keras.optimizers.Adam())

epochs = 20
batch_size = 128

vae.fit(
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None)
)

# =====================================================================
# 7. 結果の可視化 (潜在空間 & 生成画像)
# =====================================================================

# ---------------------------
# 7-1. 潜在空間の可視化
# ---------------------------
# テストセットの画像をエンコードして、(z_mean, z_log_var, z) の z_meanを2次元散布図にプロット
z_means, z_log_vars, z_points = vae.encoder.predict(x_test)
plt.figure(figsize=(8, 6))
plt.scatter(z_means[:, 0], z_means[:, 1], alpha=0.5)
plt.colorbar()
plt.title("Latent space (z_mean) distribution")
plt.xlabel("z_mean[0]")
plt.ylabel("z_mean[1]")
plt.show()

# ---------------------------
# 7-2. 潜在空間からのサンプリングで画像を生成
# ---------------------------
# -2 ~ 2 の範囲で潜在変数 z を格子状にサンプリングし、生成器(Decoder)に入力
n = 15  # 表示したいグリッドの数
figure = np.zeros((28 * n, 28 * n))  # 画像を並べて表示するためのキャンバス

grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])  # shape=(1,2)
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[i * 28 : (i + 1) * 28,
               j * 28 : (j + 1) * 28] = digit

plt.figure(figsize=(8, 8))
plt.imshow(figure, cmap="gray")
plt.title("Generated digits by sampling latent space")
plt.axis("off")
plt.show()
