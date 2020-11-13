import cv2
import numpy as np
import matplotlib.pyplot as plt


def contour_area_0():
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 复制原图
    img1 = img.copy()
    # 按照面积将所有轮廓逆序排序
    contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    i = 0
    for c in contours2:
        area = cv2.contourArea(c)
        print(area)
        i+=1
        if i>=3:break
        # 分别在复制的图像上和白色图像上绘制当前轮廓
        cv2.drawContours(img1, [c],0, (0,255,0), 3)
    cv2.imshow('image',img)
    cv2.imshow('contour',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contour_area_1():
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 复制原图
    img1 = img.copy()
    # 创建一幅相同大小的白色图像
    img2 = np.ones(img.shape)
    # 按照面积将所有轮廓逆序排序
    contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    i = 0
    for c in contours2:
        area = cv2.contourArea(c)
        print(area)
        i+=1
        if i>=3:break
        # 分别在复制的图像上和白色图像上绘制当前轮廓
        cv2.drawContours(img1, [c],0, (0,255,0), 3)
        cv2.drawContours(img2, [c],0, (0,255,0), 3)
    plot_images(1,3,[img,img1,img2], gray=True)


def plot_images(rows, cols, images, cvtColor=True, gray=False, titles=None):
    for i in range(rows*cols):
        plt.subplot(rows,cols,i+1)
        if not gray and cvtColor:
            img = cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB)
        else:
            img = images[i]
        if gray:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        if titles is not None and len(titles)>i:
            plt.title(titles[i])

    plt.show()

contour_area_0()