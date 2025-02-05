from layer_native import MulLayer

apple = 100
apple_num = 2
tax = 1.1

apple_price = apple * apple_num

ml1 = MulLayer()
ml2 = MulLayer()

apple_price = ml1.forward(apple, apple_num)
price = ml2.forward(apple_price, tax)

# print(price)

dprice = 1
dapple_price, dtax = ml2.backward(dprice)
dapple, dapple_num = ml1.backward(dapple_price)

print(f"消費税が増加した時の変化の割合{dtax} りんご1個の料金が増加した時の変化の割合 {dapple}, りんごの数が増加した時の変化の割合{dapple_num}")