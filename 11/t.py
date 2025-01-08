import numpy as np
import pandas as pd

def target_function(x, a1, a2, a3):
    """目標とする関数: a1 + a2*sin(3.1416*x/5) + a3*x"""
    return a1 + a2 * np.sin(3.1416 * x / 5) + a3 * x

def loss_function(params, x, y_true):
    """
    損失関数: 平均二乗誤差（MSE）を計算
    params: [a1, a2, a3]のリスト
    x: 入力データ
    y_true: 実際の出力値
    """
    a1, a2, a3 = params
    y_pred = target_function(x, a1, a2, a3)
    return np.mean((y_true - y_pred) ** 2)

def numerical_gradient(params, x, y_true, epsilon=1e-7):
    """
    数値微分により勾配を計算
    params: [a1, a2, a3]のリスト
    epsilon: 微小変化量
    """
    grads = np.zeros_like(params)
    for i in range(len(params)):
        # i番目のパラメータだけepsilonだけ変化させる
        params_plus = params.copy()
        params_plus[i] += epsilon
        params_minus = params.copy()
        params_minus[i] -= epsilon
        
        # 中心差分により勾配を計算
        grads[i] = (loss_function(params_plus, x, y_true) - 
                   loss_function(params_minus, x, y_true)) / (2 * epsilon)
    
    return grads

def gradient_descent(x, y_true, learning_rate=0.01, n_iterations=1000):
    """
    勾配降下法による最適化
    """
    # パラメータの初期値をランダムに設定
    params = np.random.randn(3)
    
    for i in range(n_iterations):
        # 勾配の計算
        grads = numerical_gradient(params, x, y_true)
        # パラメータの更新
        params -= learning_rate * grads
        
        if i % 100 == 0:
            loss = loss_function(params, x, y_true)
            print(f"Iteration {i}, Loss: {loss:.6f}")
    
    return params

# 使用例
if __name__ == "__main__":
    # データの生成
    # x = np.linspace(0, 10, 100)
    true_params = [1.0, 2.0, 0.5]  # 真のパラメータ
    # y_true = target_function(x, *true_params)

    df = pd.read_csv("exe11.csv")
    x = df['x'].to_numpy().reshape(-1,1)
    y_true = df['y'].to_numpy()
    
    # ノイズの追加
    y_noisy = y_true + np.random.normal(0, 0.1, y_true.shape)
    
    # 最適化の実行
    optimized_params = gradient_descent(x, y_noisy)
    print("\nOptimized parameters:", optimized_params)
    print("True parameters:", true_params)