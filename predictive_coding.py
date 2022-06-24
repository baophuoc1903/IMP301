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
    for j in range(col):
        jj = j+order
        if j == 0:
            x[:, jj] = img_encode[:, j] + (coeff[:, order-1::-1] * x[:, (jj-1)::-1]).sum(axis=1)
        else:
            x[:, jj] = img_encode[:, j] + (coeff[:, order-1::-1] * x[:, (jj-1):(jj-order-1):-1]).sum(axis=1)

    x = x[:, order:]
    return x


def encoder(img, coeff=None, order=None):
    row, col = img.shape

    if coeff is None:
        coeff = np.ones((row, col))

    p = np.zeros((row, col))

    if order is None:
        order = row

    tmp_zeros = img.copy()
    for j in range(order):
        tmp_zeros[:, :j+1] = 0
        tmp_zeros[:, j+1:] = img[:, :col-j-1]
        p = p + coeff[j] * tmp_zeros

    # print('\n\n', p, '\n\n')
    return img - np.around(p)


img = cv2.imread(r"images/Original.jpeg", 0)
encode_img = encoder(img, order=None)
decode_img = decode(encode_img.copy(), order=None)
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.imshow(encode_img, cmap='gray')
plt.subplot(133)
plt.imshow(decode_img, cmap='gray')
plt.show()
cv2.imwrite('./images/Encoder_lossless_predictive_coding.jpeg', cv2.normalize(encode_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
cv2.imwrite('./images/Decoder_lossless_predictive_coding.jpeg', decode_img.astype(int))
print((img - decode_img.astype(int)).sum())
