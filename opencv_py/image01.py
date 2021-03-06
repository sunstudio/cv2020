import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img():
    # Load an color image in grayscale
    # img = cv2.imread('./images/lena.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('images/lena.jpg', cv2.IMREAD_COLOR)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()    # 关闭所有OpenCV窗口


def plot_img():
    img = cv2.imread('images/lena.jpg')
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def plot_img2():
    img1 = cv2.imread('images/lena.jpg')
    img2 = cv2.imread('images/lena.jpg')
    img3 = cv2.imread('images/messi.jpg')
    img4 = cv2.imread('images/messi.jpg')
    plt.subplot(2,2,1)
    plt.imshow(img1)
    plt.subplot(2,2,2)
    plt.imshow(img2)
    plt.subplot(2,2,3)
    plt.imshow(img3)
    plt.subplot(2,2,4)
    plt.imshow(img4)
    plt.show()


def color_distorted():
    img = cv2.imread('images/lena.jpg')
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


if __name__ == '__main__':
    plot_img2()