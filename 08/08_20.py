import numpy as np

import matplotlib.pyplot as plt
import japanize_matplotlib

X_train = np.linspace(0,2,100)
y_train = np.sin(3.14159*X_train)

plt.scatter(X_train,y_train,color="blue")
plt.title("オリジナルデータ")
plt.show()

