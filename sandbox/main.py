import numpy as np

# 活性化関数とその導関数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 平均二乗誤差（MSE）損失関数
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mean_squared_error_derivative(y_true, y_pred):
    return -(y_true - y_pred)

# モーメンタムの更新関数
class MomentumOptimizer:
    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, gradients, parameters):
        if self.v is None:
            self.v = [np.zeros_like(g) for g in gradients]

        updated_params = []

        for i, (grad, param) in enumerate(zip(gradients, parameters)):
            self.v[i] = self.momentum * self.v[i] + grad  # モーメンタムの計算
            param_update = param - self.learning_rate * self.v[i]  # パラメータの更新
            updated_params.append(param_update)

        return updated_params

# ニューラルネットワーク
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 重みとバイアスの初期化
        self.w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        self.optimizer = MomentumOptimizer(learning_rate=0.001, momentum=0.9)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2

    def backward(self, x, y_true, y_pred):
        loss_grad = mean_squared_error_derivative(y_true, y_pred)

        dz2 = loss_grad
        dw2 = np.dot(self.a1.T, dz2) / x.shape[0]
        db2 = np.sum(dz2, axis=0, keepdims=True) / x.shape[0]

        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1) / x.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True) / x.shape[0]

        return [dw1, db1, dw2, db2]

    def update_weights(self, gradients):
        dw1, db1, dw2, db2 = gradients
        params = [self.w1, self.b1, self.w2, self.b2]
        grads = [dw1, db1, dw2, db2]
        updated_params = self.optimizer.update(grads, params)
        self.w1, self.b1, self.w2, self.b2 = updated_params

# データ生成
np.random.seed(42)
x = np.random.rand(100, 1)  # 入力データ
y = 2 * x + 1 + 0.1 * np.random.randn(100, 1)  # ラベル

# モデルの初期化
input_dim = 1
hidden_dim = 10
output_dim = 1
model = SimpleNN(input_dim, hidden_dim, output_dim)

# トレーニング
epochs = 1000
for epoch in range(epochs):
    y_pred = model.forward(x)
    loss = mean_squared_error(y, y_pred)
    gradients = model.backward(x, y, y_pred)
    model.update_weights(gradients)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# テスト
x_test = np.array([[0.5]])
y_test_pred = model.forward(x_test)
print(f"Prediction for input {x_test[0]}: {y_test_pred[0]}")
