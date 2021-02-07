import cv2
import matplotlib.pyplot as plt
import numpy as np
from util import plot_images


def bgr_filter():
    img = cv2.imread('images/messi.jpg')
    lower = np.array([0,0,100])
    upper= np.array([60,60,255])
    mask = cv2.inRange(img,lower,upper)
    res = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('res',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hsv_filter():
    img = cv2.imread('t:\\images\\001.jpg', cv2.IMREAD_COLOR)
    # img = cv2.imread('images/messi.jpg')
    b,g,r = cv2.split(img)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,40,0])
    upper = np.array([36,255,255])
    mask = cv2.inRange(img2,lower,upper)
    img_cpy = img.copy()
    res = cv2.bitwise_and(img_cpy,img_cpy,mask=mask)
    cv2.imshow('res',res)
    mask = mask - 255
    res2 = cv2.bitwise_and(img_cpy,img_cpy,mask=mask)
    cv2.imshow('res2',res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def simple_thre():
    img = cv2.imread('images/pic6.png', cv2.IMREAD_GRAYSCALE)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['original','bianry','binary inv','trunc','to zero','to zero inv']
    images =[img,thresh1,thresh2,thresh3,thresh4,thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def resize0():
    img = cv2.imread('images/messi.jpg')
    h,w,_ = img.shape
    big1 = cv2.resize(src=img,  dsize=(w//2,h//2))
    big2 = cv2.resize(src=img, dsize=None, fx=0.5, fy=0.8)
    cv2.imshow('original', img)
    cv2.imshow('big1',big1)
    cv2.imshow('big2',big2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transform0():
    img = cv2.imread('images/messi.jpg')
    rows, cols,_ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    plot_images(1,2,[img,dst])


def transform1():
    img = cv2.imread('images/messi.jpg')
    rows, cols,_ = img.shape
    src = np.float32([[0,0],[600,0],[0,450]])
    dst = np.float32([[50,20],[580,80],[80,400]])
    M = cv2.getAffineTransform(src, dst)
    dst = cv2.warpAffine(img, M, (cols, rows))
    plot_images(1,2,[img,dst])


def perspective():
    img = cv2.imread('images/perspective.jpg')
    rows, cols,_ = img.shape
    src = np.float32([[78,82],[455,70],[41,472],[455,472]])
    dst = np.float32([[41,40],[475,40],[41,472],[455,472]])
    M = cv2.getPerspectiveTransform(src, dst)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    plot_images(1,2,[img,dst])


def perspective2():
    img = cv2.imread('images/perspective.jpg')
    rows, cols, ch = img.shape
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300, 300))
    cv2.imshow('input', img)
    cv2.imshow('output', dst)

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()


if __name__ == '__main__':
    hsv_filter()
