import cv2
import numpy as np
from util import plot_images


def erode01():
    img = cv2.imread('images/j.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5), np.uint8)
    img1 = cv2.erode(img, kernel, iterations=1)
    plot_images(1,2,[img,img1])


def dilate01():
    img = cv2.imread('images/j.png')
    kernel = np.ones((5,5), np.uint8)
    img2 = cv2.dilate(img, kernel, iterations=1)
    plot_images(1,2,[img,img2])


def open01():
    img = cv2.imread('images/j1.png')
    kernel = np.ones((5,5), np.uint8)
    img1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    plot_images(1,2,[img,img1])


def morphology01():
    img = cv2.imread('images/j.png')
    kernel = np.ones((5,5), np.uint8)
    #img1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img1 = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    plot_images(1,2,[img,img1])


def canny01():
    img = cv2.imread('images/messi.jpg', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.Canny(img, 120, 220)
    plot_images(1,2,[img,img1], gray=True)


if __name__ == '__main__':
    canny01()

