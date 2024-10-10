import numpy as np
from random import uniform

def neuron_output1(x1, x2, w1, w2, theta):
    X = np.array([x1, x2])
    W = np.array([w1, w2])
    v1 = X@W
    z = 1 if v1 > theta else 0
    return z

def neuron_output2(x1, x2, x3, w1, w2, w3, theta):
    X = np.array([x1, x2, x3])
    W = np.array([w1, w2, w3])
    v1 = X@W
    z = 1 if v1 > theta else 0
    return z

def q1():
    """
    下の図のような単純パーセプトロンのネットワークがある。X1,X2は　0もしくは１とする。Hは閾値θのヘヴィサイド関数とする。
    x1, x2 には 0 か 1 かいずれかの値が入るとする。４パターンに対して出力 z が次のようになるような w1 , w2 ,θ　を１組見つけよ。
    """
    # w1:-4.119120897197872 w2:-3.2878729229562733 theta:-4.370929351061252
    
    correct_arr = np.array([0, 1, 1, 1])
    tmp_arr = np.zeros(4)
    input_arr = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    w1, w2, theta = 0, 0, 0

    while(not np.allclose(correct_arr, tmp_arr)):
        w1, w2, theta = uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)
        tmp_res = [neuron_output1(input_arr[i, 0], input_arr[i, 1], w1, w2, theta) for i in range(len(tmp_arr))]
        tmp_arr[:] = tmp_res
    print(tmp_arr)
    print(f"問1 w1:{w1} w2:{w2} theta:{theta}")

def q2():
    """
    問2 (1)
    x1, x2 には 0 か 1 かいずれかの値が入るとする。４パターンに対して出力 z が次のようになるような重み w1, w2, w3 を１組見つけよ。
    """
    # w1:2.1930129843031025 w2:2.5232814185122834 w3:-4.728870556816927
    theta = 0.55
    correct_arr = np.array([0, 1, 1, 0])
    tmp_arr = np.zeros(4)
    input_arr = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

    while(not np.allclose(correct_arr, tmp_arr)):
        w1, w2, w3 = uniform(-5, 5), uniform(-5, 5), uniform(-5, 5)
        tmp_res = [neuron_output2(input_arr[r, 0], input_arr[r, 1], input_arr[r, 0] * input_arr[r, 1], w1, w2, w3, theta) for r in range(len(tmp_arr))]
        tmp_arr[:] = tmp_res
        
    print(tmp_arr)
    print(f"問2 w1:{w1} w2:{w2} w3:{w3}")

    """
    (2) 
    入力層に x1*x2 というニューロンを加えるだけで通常の単純パーセプトロンでは不可能である XOR の演算が実現した。どうしてだろうか、あなたの考えを述べよ。
    XORはANDやORと異なり非線形であり、単純なx1やx2のみでは非線形を表現できないが、x1とx2をかけたx3を加えることで非線形の情報を扱うことができるようになるから。
    """

def main():
    q1()
    # q2()

if __name__ == "__main__":
    main()