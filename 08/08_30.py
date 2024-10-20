import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

model = MLPRegressor(
    # hidden_layer_sizes=(200,100, 100, 100),
    hidden_layer_sizes=(100, 100),
    #activation="relu",
    activation="tanh", #""tanh", #"identity"  "logistic" "relu"
    max_iter=2000,
    verbose = True,
    learning_rate_init=0.002,
    random_state=42
)

X_train = np.linspace(0,2,100)
y_train = np.sin(3.14159*X_train)

X_train = X_train.reshape(-1,1)

sc_on = True
if sc_on:
    print(X_train)
    sc = StandardScaler()
    # X_trainを標準化
    X_train = sc.fit_transform(X_train)
    print(X_train)

#y_train = y_train.reshape(-1,1)
#y_train = sc.fit_transform(y_train)
#y_train =


model.fit(X_train, y_train)
y_pred = model.predict(X_train)

print("----------------------------")

# print(model.coefs_)
# #print(model.coefs_)
# print(model.n_layers_)
# print(model.n_outputs_)
# print(model.out_activation_)
# print(model.loss_curve_)
# print(model.get_params())
# print(f"正解率：{accuracy_score(y_train, y_pred)}")

if sc_on:
    # X_trainを標準化前のデータに戻す
    X_train = sc.inverse_transform(X_train)
#y_train =sc.inverse_transform(y_train)
#y_pred = y_pred.reshape(-1,1)
#y_pred =sc.inverse_transform(y_pred)

plt.scatter(X_train,y_train,color="blue")
plt.scatter(X_train,y_pred,color="red")
plt.show()


plt.plot(model.loss_curve_)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.grid(True)
plt.show()

# モデルを保存
#filename = 'finalized_model.sav'
#pickle.dump(model, open(filename, 'wb'))

# 保存したモデル
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(model.score(X_test,y_test))
