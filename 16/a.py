import numpy as np

a = np.array([[10], [20], [30]])
b = np.array([1, 2, 3])

print(a - b.reshape(-1, 1))
print(a.flatten() - b)