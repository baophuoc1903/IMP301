import cv2
import matplotlib.pyplot as plt
import numpy as np


def decode(img_encode, coeff=None, order=None):
    row, col = img_encode.shape

    if coeff is None:
        coeff = np.ones((row, col))

    coeff = coeff[::-1, ::-1]

    if order is None:
        order = coeff.shape[0]

    x = np.zeros((row, col+order))
    print(x.shape, coeff.shape)
    for j in range(col):
        jj = j+order
        if j == 0:
            x[:, jj] = img_encode[:, j] + np.around((coeff[:, order-1::-1] * x[:, (jj-1)::-1]).sum(), decimals=2)
        else:
            x[:, jj] = img_encode[:, j] + np.around((coeff[:, order-1::-1] * x[:, (jj-1):(jj-order-1):-1]).sum(), decimals=2)

    x = x[:, order:]
    return x


def encoder(img, coeff=None, order=None):
    row, col = img.shape

    if coeff is None:
        coeff = np.ones((row, col))

    p = np.zeros((row, col))

    if order is None:
        order = row

    for j in range(order):
        tmp_zeros = np.zeros((row, col))
        tmp_zeros[:, j:] = img[:, :row-j]
        p = p + coeff[j] * tmp_zeros

    return img - np.around(p)


img = cv2.imread(r"images/Original.jpeg", 0)
encode_img = encoder(img, order=2)
decode_img = decode(encode_img.copy(), order=2)
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.imshow(encode_img, cmap='gray')
plt.subplot(133)
plt.imshow(decode_img, cmap='gray')
print(encode_img, end='\n\n\n\n\n')
print(decode_img)
plt.show()

