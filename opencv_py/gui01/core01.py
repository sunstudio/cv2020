import numpy as np
import cv2
from matplotlib import pyplot as plt
from operator import mul
from functools import reduce


def pixel():
    # Load an color image in grayscale
    # img = cv2.imread('./images/lena.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('../images/lena.jpg')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    img[150:,250:,:]=0
    plt.imshow(img)
    p = img[100,200]
    print(p)
    blue = img[100,200,0]
    print(blue)
    plt.show()


def size():
    img = cv2.imread('../images/messi.jpg', cv2.IMREAD_COLOR)
    print(img.shape)
    print(img.size)
    print(reduce(mul, img.shape))
    print(img.dtype)


size()