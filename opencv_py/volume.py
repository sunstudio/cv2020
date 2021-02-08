import cv2
import matplotlib.pyplot as plt
import numpy as np
from util import plot_images


def hsv_values():
    # files = ['009','010','011','b','g','r','black']
    # names = ['box','green','red','b','g','r','black']
    files = ['009','010','011','b1','b2']
    names = ['box','green','red','b1','b2']
    for i in range(len(files)):
        print('file: ', names[i])
        img = cv2.imread(f't:\\images\\{files[i]}.jpg')
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg1=[]
        avg2=[]
        for j in range(3):
            avg1.append(round(np.average(img[:,:,j]),1))
            avg2.append(round(np.average(img2[:,:,j]),1))
        print('bgr', avg1, end='\t')
        print('hsv',avg2, end='\t')
        print()


def hsv_filter():
    img = cv2.imread('t:\\images\\001.jpg')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,50,100])
    upper = np.array([5,255,255])
    mask = cv2.inRange(img2,lower,upper)

    lower = np.array([170,50,100])
    upper = np.array([200,255,255])
    mask2 = cv2.inRange(img2,lower,upper)

    mask3 = np.bitwise_or(mask.astype(np.bool), mask2.astype(np.bool))
    mask3 = mask3.astype(np.uint8)
    img_cpy = img.copy()
    res = cv2.bitwise_and(img_cpy,img_cpy,mask=mask3)

    cv2.imshow('res',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_ruler_size():
    img = cv2.imread('t:\\images\\r-057.jpg')
    img_cpy = img.copy()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0,50,100])
    upper = np.array([5,255,255])
    mask = cv2.inRange(img2,lower,upper)

    lower = np.array([170,20,80])
    upper = np.array([200,255,255])
    mask2 = cv2.inRange(img2,lower,upper)

    mask3 = np.bitwise_or(mask.astype(np.bool), mask2.astype(np.bool))
    mask3 = mask3.astype(np.uint8)
    contours,hierarchy = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    contours2 = [ (cv2.contourArea(a), a) for a in contours]
    contours2 = sorted(contours2, key = lambda a: a[0], reverse=True )
    res = cv2.bitwise_and(img_cpy,img_cpy,mask=mask3)
    minx, maxx, miny, maxy = 9999, 0, 9999, 0
    i = 0
    for con in contours2:
        area = con[0]
        c = con[1]
        if area < 1200: break
        print(area)
        cv2.drawContours(res, [c],0, (0,255,0), 3)
        minx = min(minx, np.min(c[:,:,0]))
        maxx = max(maxx, np.max(c[:,:,0]))
        miny = min(miny, np.min(c[:,:,1]))
        maxy = max(maxy, np.max(c[:,:,1]))
        i+=1
        if i>=5:break

    ruler_height = maxy - miny
    ruler_width = maxx - minx
    print('ruler', ruler_height, ruler_width)
    cv2.imshow('img', img)
    cv2.imshow('res',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_old_ruler_size():
    img = cv2.imread('t:\\images\\r-043.jpg')
    img_cpy = img.copy()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([90,50,100])
    upper = np.array([100,255,255])
    mask = cv2.inRange(img2,lower,upper)

    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    contours2 = [ (cv2.contourArea(a), a) for a in contours]
    contours2 = sorted(contours2, key = lambda a: a[0], reverse=True )
    res = cv2.bitwise_and(img_cpy,img_cpy,mask=mask)
    minx, maxx, miny, maxy = 9999, 0, 9999, 0
    i = 0
    for con in contours2:
        area = con[0]
        c = con[1]
        if area < 1200: break
        print(area)
        cv2.drawContours(res, [c],0, (0,255,0), 3)
        minx = min(minx, np.min(c[:,:,0]))
        maxx = max(maxx, np.max(c[:,:,0]))
        miny = min(miny, np.min(c[:,:,1]))
        maxy = max(maxy, np.max(c[:,:,1]))
        i+=1
        if i>=5:break

    ruler_height = maxy - miny
    ruler_width = maxx - minx
    print('ruler', ruler_height, ruler_width)
    cv2.imshow('img', img)
    cv2.imshow('res',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_box_size():
    img = cv2.imread('t:\\images\\r-057.jpg')
    img_cpy = img.copy()
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([11,20,80])
    upper = np.array([20,255,255])
    mask = cv2.inRange(img2,lower,upper)
    contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours2 = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
    contours2 = [ (cv2.contourArea(a), a) for a in contours]
    contours2 = sorted(contours2, key = lambda a: a[0], reverse=True )
    res = cv2.bitwise_and(img_cpy,img_cpy,mask=mask)
    minx, maxx, miny, maxy = 9999, 0, 9999, 0
    i = 0
    for con in contours2:
        area = con[0]
        c = con[1]
        if area<2500: break
        print(area)
        minx0 = np.min(c[:,:,0])
        maxx0 = np.max(c[:,:,0])
        miny0 = np.min(c[:,:,1])
        maxy0 = np.max(c[:,:,1])
        # 忽略细长条
        if maxx0 - minx0 < 40 or maxy0 - miny0 < 40:
            continue
        minx = min(minx, minx0)
        miny = min(miny, miny0)
        maxx = max(maxx, maxx0)
        maxy = max(maxy, maxy0)
        cv2.drawContours(res, [c],0, (0,255,0), 3)
        i+=1
        if i>=5:break

    ruler_height = maxy - miny
    ruler_width = maxx - minx
    print('box', ruler_height, ruler_width)
    cv2.imshow('img', img)
    cv2.imshow('res',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # hsv_values()
    # hsv_filter()
    # get_ruler_size()
    get_old_ruler_size()
    # get_box_size()