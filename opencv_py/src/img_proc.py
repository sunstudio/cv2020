import cv2
import matplotlib.pyplot as plt
import numpy as np

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

simple_thre()

