import math
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

"""
Himmelblau関数
local max =-0.270845 and y = -0.923039

f(3.0,2.0)=0.0
f(-2.805118,3.131312)=0.0
f(-3.779310,-3.283186)=0.0
f(3.584428,-1.848126)=0.0
"""

# 数値微分の関数 f はベクトルvec_wの関数である。
def get_grad(f, vec_w, h=0.0001):
    grad = np.zeros_like(vec_w)
    for i in range(len(vec_w)):
        vec_i_org = vec_w[i]
        vec_w[i] = vec_i_org + h
        fh1 = f(vec_w)
        vec_w[i] = vec_i_org - h
        fh2 = f(vec_w)
        grad[i] = (fh1 - fh2) / (2 * h)
        vec_w[i] = vec_i_org
    return grad

def func_01(w):
    x = w[0]
    y = w[1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# 学習率とモーメンタム係数の設定
eta = 0.005
batch_size = 5  # バッチサイズを設定
w_list = [np.array([-4.0, -1.7]), np.array([0.5, 1.5]), np.array([-3.0, 2.0]), np.array([2.0, -2.0]), np.array([4.0, 4.0])]  # 初期点のリスト

# 各初期点に対応するモーメンタムの初期化
v_list = [np.zeros_like(w) for w in w_list] #[array([0., 0.]), array([0., 0.]), array([0., 0.]), array([0., 0.]), array([0., 0.])]
print(v_list)
# for epoch in range(1, 501):
#     total_grad = np.zeros_like(w_list[0])
    
#     # バッチごとに勾配を計算して平均を取る
#     for i in range(batch_size):
#         grad = get_grad(func_01, w_list[i], h=0.0001)
#         total_grad += grad
    
#     total_grad /= batch_size  # 勾配の平均を計算
    
#     # バッチ全体の重みを更新
#     for i in range(batch_size):
#         v_list[i] = eta * total_grad  # 勾配の平均を用いてモーメンタム更新
#         w_list[i] = w_list[i] - v_list[i]  # パラメータの更新
    
#     # 停止条件の判定
#     if np.linalg.norm(total_grad, ord=2) < 0.001:
#         break
    
#     # 各エポックの出力
#     print(f"epoch:{epoch}, w_list={w_list}", end=" ")
#     print(f"E={func_01(w_list[0])}")  # 1つ目のバッチのEを表示
#     plt.scatter(epoch, func_01(w_list[0]), color="green")

# plt.xlabel("学習回数")
# plt.ylabel("関数Eの値")
# plt.show()
