import numpy as np

def heavyWeightFun(x, w, theta):
 xw = np.dot(x, w)
 if(xw >= theta):
  return 1
 else:
  return 0
 

def main():
 X = np.array([[1, 1], [1, 0], [0, 1], [0, 1]])
 w = np.array([1, 1])
 theta = 2

 for x in X:
  z = heavyWeightFun(x, w, theta)
  print(f"x = {x}, w = {w}, z = {z}")


if __name__ == "__main__":
 main()