import cv2
import numpy as np


def knn_class():
    # read the big image (containing 100*50 small digit images)
    for i in range(1,7):
        img = cv2.imread('../images/hair/head{0:03d}.jpg'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32))
        x = np.array(gray).astype(np.float32).reshape(-1, 32 * 32)
        if i == 1:
            train = x
            labels = np.array([[1]])
        else:
            train = np.vstack((train,x))
            labels = np.vstack((labels, [1]))
        # print(x.shape)
    for i in range(1,7):
        img = cv2.imread('../images/helmet/hat{0:03d}.jpg'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32))
        train = np.vstack((train, np.array(gray).astype(np.float32).reshape(-1, 32*32)))
        labels = np.vstack((labels, [2]))
    print('shape')
    print(train.shape)
    print(labels.shape)
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, labels)
    # read the test image
    img = cv2.imread('images/helmet/hat007.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('../images/hair/head007.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32,32))
    test = np.array(img).astype(np.float32).reshape(-1,32*32)
    # use knn to find nearest neighbors
    ret,result,neighbours,dist = knn.findNearest(test, k=3)
    print(result)
    print(neighbours)
    print(dist)


def svm_class():

    # 分别读取头盔和头发前6幅图(只做了这么多图），作为训练样本

    # 读取头发（无安全帽）图片
    for i in range(1,7):
        img = cv2.imread('../images/hair/head{0:03d}.jpg'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32))
        x = np.array(gray).astype(np.float32).reshape(-1, 32 * 32)
        if i == 1:
            train = x
            labels = np.array([[1]])
        else:
            train = np.vstack((train,x))
            labels = np.vstack((labels, [1]))

    # 读取安全帽图片
    for i in range(1,7):
        img = cv2.imread('../images/helmet/hat{0:03d}.jpg'.format(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32))
        train = np.vstack((train,
                           np.array(gray).astype(np.float32).reshape(-1, 32*32)))
        labels = np.vstack((labels, [2]))

    # 创建svm并设置参数
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(0.1)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e5), 1e-6))

    # 训练
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    print('finished training process')

    # 读入测试图片
    # img = cv2.imread('../images/helmet/hat007.jpg', cv2.IMREAD_COLOR)
    img = cv2.imread('images/hair/head007.jpg', cv2.IMREAD_COLOR)
    test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test = cv2.resize(test, (32,32))
    test = np.array(test).astype(np.float32).reshape(-1,32*32)
    # 用前面训练的svm进行预测
    result = svm.predict(test)[1]
    name = 'hair' if result == 1 else 'helmet'
    # 输出结果，并在图上显示结果
    print(name)
    cv2.putText(img, name, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,0))
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


svm_class()