# importing libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.layers import MaxPool2D, Activation, MaxPooling2D

# loading dataset
cifar = keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar.load_data()

# defining labels
label = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# rescaling data between 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

# plotting image
plt.figure()
plt.imshow(X_train[26])
plt.colorbar()

# buiding model
model = Sequential()
model.add(Conv2D(64, activation='relu', kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, activation='relu', kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(80, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compilation of  model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fitting model
history = model.fit(X_train, y_train, epochs=10, verbose=1, validation_split=0.3)

# plotting losses
def plotloss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'accuracy'], loc='lower right')
    plt.show()

plotloss(history)