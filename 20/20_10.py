from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

#画像ファイルをnumpyに変換する
img = Image.open('hinton.jpg')
img = np.array(img)
print(img)

##img = cv2.imread('image.png')
# グレースケールにする
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
print(img_gray)

plt.imshow(img_gray, cmap="gray")
plt.show()

# フィルター
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

img_conv = cv2.filter2D(img_gray, -1, kernel)
plt.imshow(img_conv, cmap='gray');
plt.savefig("cnn.jpg")
plt.show()

