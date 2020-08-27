import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img():
    # Load an color image in grayscale
    # img = cv2.imread('./images/lena.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_COLOR)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()    # 关闭所有OpenCV窗口


def plot_img():
    img = cv2.imread('../images/lena.jpg')
    plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def color_distorted():
    img = cv2.imread('../images/lena.jpg')
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


if __name__ == '__main__':
    color_distorted()