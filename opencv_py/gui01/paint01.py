import numpy as np
import cv2


def draw_line():
    # Create a black image
    img = np.zeros((600, 800, 3))
    # Draw a diagonal blue line with thickness of 5 px
    img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_line2():
    img = np.zeros((300, 400, 3))
    left = (10,290)
    right = (390,290)
    top = (200,10)
    img = cv2.line(img, left, right, (0, 0, 255), 3)
    img = cv2.line(img, left, top, (0, 0, 255), 3)
    img = cv2.line(img, top, right, (0, 0, 255), 3)

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_line3():
    img = cv2.imread('../images/lena.jpg')
    # Draw a diagonal blue line with thickness of 5 px
    img = cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_rect():
    img = cv2.imread('../images/lena.jpg')
    cv2.rectangle(img,(80,50), (400,400), (0,255,255), 3)
    cv2.circle(img, (240, 225), 180, (0, 0, 255), 2)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_circle():
    img = np.zeros((300, 400, 3), np.uint8)
    center = (200,150)
    for i in range(3):
        r = 140*(3-i)//3
        c = 70 *(i+1)
        cv2.circle(img, center, r, (150,80,c), -1)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_text():
    img = cv2.imread('../images/lena.jpg')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,"Lina", (150,80), font, 4, (100,50,0), 2)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


draw_text()