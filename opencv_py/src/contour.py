import cv2
import numpy as np
from util import plot_images


def contour00():
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img1 = img.copy()
    cv2.drawContours(img1, contours, -1, (0,255,0), 3)
    plot_images(1,2,[img,img1], gray=True)


def contour_area():
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 复制原图
    img1 = img.copy()
    # 创建一幅相同大小的白色图像
    img2 = np.ones(img.shape)
    # 按照面积将所有轮廓逆序排序
    contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    for c in contours2:
        area = cv2.contourArea(c)
        print(area)
        # 只输出面积大于500轮廓
        if area<500:break
        # 分别在复制的图像上和白色图像上绘制当前轮廓
        cv2.drawContours(img1, [c],0, (0,255,0), 3)
        cv2.drawContours(img2, [c],0, (0,255,0), 3)
    plot_images(1,3,[img,img1,img2], gray=True)


def contour_perimeter():
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img1 = img.copy()
    img2 = np.ones(img.shape)
    # 按照弧长将所有轮廓逆序排序
    contours2 = sorted(contours, key=lambda a: cv2.arcLength(a, False), reverse=True)
    i = 0
    for c in contours2:
        l = cv2.arcLength(c, False)
        print(l)
        # 只输出弧长大于100的轮廓
        if l<100: break
        # 分别在复制的图像上和白色图像上绘制当前轮廓
        cv2.drawContours(img1, [c],0, (0,255,0), 3)
        cv2.drawContours(img2, [c],0, (0,255,0), 3)
    plot_images(1,3,[img,img1,img2], gray=True)


def moments01():
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    # 二值化
    ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 按照轮廓弧长排序
    contours2 = sorted(contours, key=lambda a: cv2.arcLength(a, False), reverse=True)
    # 找到最大弧长的轮廓
    cnt = contours2[0]
    # 计算图像矩
    m = cv2.moments(contours2[0])
    # 输出图像矩
    print(m)
    # 输出图像矩的所有键
    for k in m:
        print(k, end=',')
    # 创建一个白色背景图片
    img2 = np.ones(img.shape)
    # 在其上绘制前面找到的最长的那个轮廓
    # 以下2种写法等价
    # cv2.drawContours(img2,contours2,0,(0,255,0),3)
    cv2.drawContours(img2,[cnt],0,(0,255,0),3)
    # 按照公式计算中心
    center = (int(m['m10']/m['m00']),int(m['m01']/m['m00']))
    # 在中心点绘制一个小实心圆
    cv2.circle(img2, center, 20, (0,255,0), -1)
    # 用matplotlib显示这2幅图
    plot_images(1,2,[img,img2], gray=True)


def approximation01():
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    # 二值化
    ret,thresh = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
    # 查找轮廓
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 将轮廓按照弧长排序
    contours2 = sorted(contours, key=lambda a: cv2.arcLength(a, False), reverse=True)
    # 取最长的轮廓
    cnt = contours2[0]
    # 绘制最长轮廓
    img2 = np.ones(img.shape)
    cv2.drawContours(img2,[cnt],0,(0,255,0),3)

    # epsilon取0.01倍弧长来近似轮廓并绘制
    epsilon = 0.01*cv2.arcLength(cnt,True)
    cnt1 = cv2.approxPolyDP(cnt,epsilon,True)
    img3 = np.ones(img.shape)
    cv2.drawContours(img3,[cnt1],0,(0,255,0),3)

    # epsilon取0.03倍弧长来近似轮廓并绘制
    epsilon = 0.03*cv2.arcLength(cnt,True)
    cnt2 = cv2.approxPolyDP(cnt,epsilon,True)
    img4 = np.ones(img.shape)
    cv2.drawContours(img4,[cnt2],0,(0,255,0),3)

    plot_images(2,2,[img,img2,img3,img4], gray=True)


def convex01():
    img = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)
    # 二值化
    ret,thresh = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
    # 查找轮廓
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 将轮廓按照弧长排序
    contours2 = sorted(contours, key=lambda a: cv2.arcLength(a, False), reverse=True)
    # 取最长的轮廓
    cnt = contours2[0]
    # 绘制最长轮廓
    img2 = np.ones(img.shape)
    cv2.drawContours(img2,[cnt],0,(0,255,0),3)
    # 取最长轮廓的凸外壳
    hull = cv2.convexHull(cnt)
    # 绘制凸外壳
    cv2.drawContours(img2,[hull],0,(0,255,0),3)
    plot_images(1,2,[img,img2], gray=True)


def boundingRectangle():
    img = cv2.imread('../images/lightning.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    # direct rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
    # rotated rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),3)
    cv2.imshow('preview',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# approximation01()
boundingRectangle()