import cv2
import numpy as np
import matplotlib.pyplot as plt


def knn_rand_plot():
    number = 30
    trainingData = np.random.randint(0,100,(number,2)).astype(np.float32)
    responses = np.random.randint(0,2,(number,1)).astype(np.float32)
    responses = responses.ravel()
    red = trainingData[responses == 0]
    green = trainingData[responses == 1]
    # print(red)
    # print(red[:,0])
    # print(red[:,1])
    plt.scatter(red[:,0],red[:,1],30,'r','^')
    plt.scatter(green[:,0],green[:,1],30,'b','s')

    newcomer = np.random.randint(0,100,(3,2)).astype(np.float32)
    # newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
    plt.scatter(newcomer[:,0],newcomer[:,1],30,'g','o')
    knn=cv2.ml.KNearest_create()
    knn.train(trainingData, cv2.ml.ROW_SAMPLE, responses)
    ret, results, neighbours, dist = knn.findNearest(newcomer, 5)
    print(f'result:{results}')
    print(f'neighbour:{neighbours}')
    print(f'distance:{dist}')
    plt.show()


def knn_digits_ocr():
    img = cv2.imread('../images/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = []
    # now we split the image to 5000 cells, each 20x20 size
    for i in range(50):
        row = gray[i*20:(i+1)*20,:]
        for j in range(100):
            cell = row[:,j*20:(j+1)*20]
            cells.append(cell)
    cells = np.array(cells).reshape((50,100,20,20))
    # cv2.destroyAllWindows()
    x = np.array(cells)
    # take first 50 sample as train dataset and reshape to (2500,400)
    train = x[:,:50].reshape(-1,400).astype(np.float32)
    # take last 50 samples as test dataset and reshape to (2500,400)
    test = x[:,50:100].reshape(-1,400).astype(np.float32)

    # create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,250).reshape((2500,1))
    test_labels = train_labels.copy()

    # initialize KNN, train it to the training data, then test it with the test data with k=1
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    # now we check the accuracy of classification by comparing the result with test_labels
    matches = result == test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print('accuracy is ', accuracy)


def recognize():
    # read the big image (containing 100*50 small digit images)
    img = cv2.imread('../images/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # split into 100*50 cells, each cell is a digit image
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)
    # convert each image into 20*20=400 array (vector)
    train = x.reshape(-1,400).astype(np.float32)
    # generate labels for the training data
    k = np.arange(10)
    train_labels=np.repeat(k,500).reshape(5000,1)
    print(train.shape)
    print(train_labels.shape)
    # create knn and train
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    # read the test image
    img = cv2.imread('../images/digits/7.png', cv2.IMREAD_GRAYSCALE)
    test = np.array(img).astype(np.float32).reshape(-1,400)
    # use knn to find nearest neighbors
    ret,result,neighbours,dist = knn.findNearest(test, k=5)
    print(result)
    print(neighbours)
    print(dist)


def plot_test():
    # 前两列是数据，后一列是标签（类别）
    dataset = [[1.2,5.0,1],[1.8,6.0,1],
               [2.1,5.8,1],[8.2,1.5,2],
               [7.8,5.3,2],[2.3,4.4,1],
               [9.2,3.8,2]]
    dataset = np.array(dataset)
    for item in dataset:
        x = item[0]
        y = item[1]
        l = item[2]
        c = 'b' if l==1 else 'r'
        m = '^' if l==1 else 's'
        plt.scatter(x, y, c=c, s=18, marker=m)
    plt.show()
    print(x)


def predicate():
    dataset = [[1.2,5.0,1],[1.8,6.0,1],
               [2.1,5.8,1],[8.2,1.5,2],
               [7.8,5.3,2],[2.3,4.4,1],
               [9.2,3.8,2]]
    dataset = np.array(dataset).astype(np.float32)
    label = dataset[:,-1]
    data = dataset[:,0:2]
    for item in dataset:
        x = item[0]
        y = item[1]
        l = item[2]
        c = 'b' if l==1 else 'r'
        m = '^' if l==1 else 'o'
        plt.scatter(x, y, c=c, s=12, marker=m)
    point = np.array([[6.0, 4.2]]).astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(data, cv2.ml.ROW_SAMPLE, label)
    ret, results, neighbours, dist = knn.findNearest(point, 3)
    print(results)
    print(neighbours)
    print(dist)
    plt.scatter(point[0][0],point[0][1],c='g', s=18, marker='o')
    plt.show()


recognize()
# knn_rand_plot()
# predicate()