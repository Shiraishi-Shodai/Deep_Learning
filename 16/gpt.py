import numpy as np

# シグモイド関数（活性化関数）とその微分
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # シグモイドの微分: f'(x) = f(x) * (1 - f(x))

# データセット（X: 入力データ, y: 正解ラベル）
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 2入力（ANDのようなデータ）
y = np.array([[0], [1], [1], [0]])  # XOR問題

# 重みとバイアスの初期化
np.random.seed(0)  # 乱数固定（再現性のため）
input_size = 2   # 入力層のニューロン数
hidden_size = 3  # 隠れ層のニューロン数
output_size = 1  # 出力層のニューロン数

# 重み（ランダム初期化）
W1 = np.random.randn(input_size, hidden_size)  # 入力層 → 隠れ層
b1 = np.zeros((1, hidden_size))  # 隠れ層のバイアス

W2 = np.random.randn(hidden_size, output_size)  # 隠れ層 → 出力層
b2 = np.zeros((1, output_size))  # 出力層のバイアス

# 学習率
learning_rate = 0.1
epochs = 1  # 学習回数

# 学習ループ
for epoch in range(epochs):
    # 順伝播（Forward Propagation）
    hidden_input = np.dot(X, W1) + b1  # 隠れ層への入力
    hidden_output = sigmoid(hidden_input)  # 隠れ層の出力
    
    final_input = np.dot(hidden_output, W2) + b2  # 出力層への入力
    final_output = sigmoid(final_input)  # 出力層の出力

    # 損失の計算（MSE: Mean Squared Error）
    loss = np.mean((y - final_output) ** 2)

    # 逆伝播（Backpropagation）
    # 出力層の誤差
    error_output = (final_output - y)  # MSEの微分
    delta_output = error_output * sigmoid_derivative(final_output)  # チェインルール

    # 隠れ層の誤差
    error_hidden = delta_output.dot(W2.T)  # 出力層からの誤差伝播
    delta_hidden = error_hidden * sigmoid_derivative(hidden_output)  # チェインルール

    # 重みとバイアスの更新
    W2 -= learning_rate * hidden_output.T.dot(delta_output)
    b2 -= learning_rate * np.sum(delta_output, axis=0, keepdims=True) # 各項の誤差の合計に学習率をかける

    W1 -= learning_rate * X.T.dot(delta_hidden)
    b1 -= learning_rate * np.sum(delta_hidden, axis=0, keepdims=True) # 各項の誤差の合計に学習率をかける

    # 1000エポックごとに損失を表示
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.5f}")

# 学習後の予測
print("\n学習後の予測:")
print(sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2))
