import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_images(rows, cols, images, cvtColor=True, gray=False, titles=None):
    for i in range(rows*cols):
        if i>=len(images): break
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

