import numpy as np
import cv2
from matplotlib import pyplot as plt
from operator import mul
from functools import reduce
from util import plot_images


def pixel():
    # Load an color image in grayscale
    # img = cv2.imread('./images/lena.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('images/lena.jpg')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    img[150:,250:,:]=0
    plt.imshow(img)
    p = img[100,200]
    print(p)
    blue = img[100,200,0]
    print(blue)
    plt.show()


def size():
    img = cv2.imread('images/messi.jpg', cv2.IMREAD_COLOR)
    print(img.shape)
    print(img.size)
    print(reduce(mul, img.shape))
    print(img.dtype)


def split():
    # img = cv2.imread('../images/messi.jpg', cv2.IMREAD_COLOR)
    img = cv2.imread('t:\\images\\001.jpg', cv2.IMREAD_COLOR)
    b,g,r = cv2.split(img)
    # b3 = np.repeat(b,3).reshape(img.shape)
    # img -= b3
    # img *= (img<250)
    g2 = (g - r)
    # g2 *= (g2<250)
    r2 = (r - g)
    # r2 *= (r2<250)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plot_images(2,2,[img,b,g2,r2], gray=True, titles=[' ', 'b', 'g', 'r'])


def border():
    BLUE = [255, 0, 0]
    img1 = cv2.imread('images/opencv-logo.png')
    replicate = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=BLUE)
    plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
    plt.show()


def blend():
    e1 = cv2.getTickCount()
    img1 = cv2.imread('images/lena.jpg')
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    plt.subplot(131)
    plt.imshow(img1)
    img2 = cv2.imread('images/opencv-logo.png')
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    plt.subplot(132)
    plt.imshow(img2)
    h,w,c=img2.shape
    roi = img1[0:h,0:w,:]
    alpha = 0.7
    img3 = cv2.addWeighted(roi,alpha, img2, (1-alpha), 0)
    img1[0:h,0:w,:]=img3
    plt.subplot(133)
    plt.imshow(img1)
    e2=cv2.getTickCount()
    time = (e2-e1)/cv2.getTickFrequency()
    print(time)
    plt.show()


def add():
    # Load two images
    img1 = cv2.imread('images/messi.jpg')
    img2 = cv2.imread('images/opencv-logo.png')
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    split()
