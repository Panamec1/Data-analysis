from platform import python_version
import cv2
import torch
import sys
import torchvision

import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena.png')

plt.figure(figsize=(15, 5))
plt.title('Lena')
plt.imshow(img)
plt.axis('off')
plt.show()

# Контвертируем изображение в нужный формат
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# "BGR2RGB" -> "Blue Green Red to Red Green Blue"

plt.figure(figsize=(15, 5))
plt.title('Lena RGB')
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_copy = img_gray.copy()

for i in range(img_gray_copy.shape[0]):  # По строкам
    for j in range(img_gray_copy.shape[1]): # По столбцам
        if i < 150 and j < 100:
            img_gray_copy[i, j] = 0

plt.figure(figsize=(15, 5))
plt.title('Lena grayscale changed')
plt.imshow(img_gray_copy, cmap='gray')
plt.axis('off');

operatedImage = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
operatedImage = np.float32(operatedImage)

dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)
dest = cv2.dilate(dest, None)
img0[dest > 0.01 * dest.max()]=[0, 0, 255]
  

print(img)
#cv2.imshow('Image with Borders', img)
ax = plt.subplot(121)
ax.set_title('Original image')
ax.axis('off')
plt.imshow(img_rgb)

ax = plt.subplot(122)
ax.set_title('Detected edges')
ax.axis('off')
plt.imshow(img0)
plt.show()



kernel = np.array([[1, 1, 1, 1, 1], [1, 2, 1, 2, 1], [2, 1, 2, 1, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

kernel = kernel/sum(kernel)
img_rst = cv2.filter2D(img0,-1,kernel)

ax = plt.subplot(121)
ax.set_title('Original image')
ax.axis('off')
plt.imshow(img0)

ax = plt.subplot(122)
ax.set_title('Detected edges')
ax.axis('off')
plt.imshow(img_rst)
plt.show()

















