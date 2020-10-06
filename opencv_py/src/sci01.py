import numpy as np
import cv2
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

def demo01():
    img = cv2.imread('../images/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]


demo01()