import tensorflow as tf
import numpy as np

# サンプルデータ
data = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [13, 14, 15, 16]])
data = data.reshape(1, 4, 4, 1)  # (batch_size, height, width, channels)

# 最大プーリング
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(data)
print("Max Pooling Result:\n", max_pool.numpy())
