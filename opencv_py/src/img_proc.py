import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.util import plot_images


def bgr_filter():
    img = cv2.imread('../images/messi.jpg')
    lower = np.array([0,0,100])
    upper= np.array([60,60,255])
    mask = cv2.inRange(img,lower,upper)
    res = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('res',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hsv_filter():
    img = cv2.imread('../images/messi.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([-15,50,50])
    upper= np.array([15,255,255])
    mask = cv2.inRange(img,lower,upper)
    res = cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow('res',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def simple_thre():
    img = cv2.imread('../images/pic6.png', cv2.IMREAD_GRAYSCALE)
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
    img = cv2.imread('../images/messi.jpg')
    h,w,_ = img.shape
    big1 = cv2.resize(src=img,  dsize=(w//2,h//2))
    big2 = cv2.resize(src=img, dsize=None, fx=0.5, fy=0.8)
    cv2.imshow('original', img)
    cv2.imshow('big1',big1)
    cv2.imshow('big2',big2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transform0():
    img = cv2.imread('../images/messi.jpg')
    rows, cols,_ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    plot_images(1,2,[img,dst])


def transform1():
    img = cv2.imread('../images/messi.jpg')
    rows, cols,_ = img.shape
    src = np.float32([[0,0],[600,0],[0,450]])
    dst = np.float32([[50,20],[580,80],[80,400]])
    M = cv2.getAffineTransform(src, dst)
    dst = cv2.warpAffine(img, M, (cols, rows))
    plot_images(1,2,[img,dst])


def perspective():
    img = cv2.imread('../images/perspective.jpg')
    rows, cols,_ = img.shape
    src = np.float32([[78,82],[455,70],[41,472],[455,472]])
    dst = np.float32([[41,40],[475,40],[41,472],[455,472]])
    M = cv2.getPerspectiveTransform(src, dst)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    plot_images(1,2,[img,dst])


def blur01():
    img0 = cv2.imread('../images/lena.jpg')
    img1 = cv2.blur(img0, (5,5))
    img2 = cv2.GaussianBlur(img0,(5,5), 0)
    img3 = cv2.medianBlur(img0,5)
    plot_images(2,2,[img0,img1,img2,img3], titles=['original','averaging','Gaussian','median'])

blur01()

