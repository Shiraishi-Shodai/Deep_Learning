import tensorflow as tf
import pandas as pd
import numpy as np
from sympy import isprime

"""
データを作る
2以上100以下のデータを２，３，５，７，１１で割ったあまりと素数かどうかの判定(0または1)。
"""

# 初期化したdfを生成
def init_df(train_scope, divisor_data):
    columns = np.zeros(len(divisor_data))
    columns = [f"{i}の剰余" for i in divisor_data]
    columns.append("素数=1/合成数=0")
    index = train_scope
    return pd.DataFrame(columns=columns, index=index)

# データフレーム1行分のデータを生成する
def make_df_row(i, divisor_data):
    rowData = list(map(lambda x: i % x, divisor_data))
    if isprime(i):
        rowData.append(1)
    else:
        rowData.append(0)
    return rowData

# 学習用、またはテスト用のデータを生成する
def makeData(train_scope, divisor_data):
    df = init_df(train_scope, divisor_data)
    for i in df.index:
        df.loc[i, :] = make_df_row(i, divisor_data)
    return df

def main():
    # 学習対象のデータ範囲
    train_scope = np.arange(2, 101)
    # テスト対象のデータ範囲
    test_scope = np.arange(101, 200)
    # 割る数の配列
    divisor_data = np.array([2, 3, 5, 7, 11])

    # 学習用とテスト用のデータフレームを生成
    train_df = makeData(train_scope, divisor_data)
    test_df = makeData(test_scope, divisor_data)
    # print(train_df.head())
    # print(test_df.head())

    # 説明変数と目的変数に学習用とテスト用でそれぞれ分割
    X_train, y_train = train_df.iloc[:, :-1].to_numpy(dtype=np.float32), train_df.iloc[:, -1].to_numpy(dtype=np.float32)
    X_test, y_test = test_df.iloc[:, :-1].to_numpy(dtype=np.float32), test_df.iloc[:, -1].to_numpy(dtype=np.float32)
    
    # モデルの構築
    model = tf.keras.models.Sequential([
        tf.keras.Input(5),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=100)
    # model.evaluate(X_test, y_test, verbose=2)
    y_pred = model.predict(X_test)
    # 0か1に変換
    binary_pred = tf.round(y_pred)

    # 素数と判定した数を格納するリスト
    prime_array = []
    for i, label in zip(test_scope, binary_pred):
        if label == 1:
            prime_array.append(i)

    print(f"素数リスト: {prime_array}")
    print(test_df.head(10))

    # モデルの保存
    model.save("prime_classter.keras")

if __name__ == "__main__":
    main()