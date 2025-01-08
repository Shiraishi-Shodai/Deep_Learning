"""
最大プーリング(ここでは画像の圧縮は行っていません)
"""
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


#画像ファイルをnumpyに変換する
img = cv2.imread('person.png')
#img = np.array(img)
#print(img)

def maxPooling(img,k):
  dst = img.copy()
  print(img.shape)
  w,h,c = img.shape # (512, 512, 3)
  # 中心画素から両端画素までの長さ
  size = k // 2
  print(size, w, h) # (20 512 512)

  # プーリング処理
  for x in range(size, w, k): # 20~511まで、40ずつ
    for y in range(size, h, k): # 20~511まで、40ずつ
      dst[x-size:x+size,y-size:y+size,0] = np.max(img[x-size:x+size,y-size:y+size,0])
      dst[x-size:x+size,y-size:y+size,1] = np.max(img[x-size:x+size,y-size:y+size,1])
      dst[x-size:x+size,y-size:y+size,2] = np.max(img[x-size:x+size,y-size:y+size,2])

  return dst

img = maxPooling(img,40)
print(img[:, :, 0])
print(img.shape) # (512, 512, 3)
cv2.imwrite('result.jpg', img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
