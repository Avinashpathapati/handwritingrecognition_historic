# CNN module
# Author: Andreas Pentaliotis
# Convolutional neural network implementation.

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.constraints import maxnorm
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras import backend as K
K.set_image_dim_ordering("tf")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


class CNN():
  def __init__(self, height, width, depth, classes):
    self.height = height
    self.width = width
    self.depth = depth
    self.classes = classes
    self.__build()

  def __build(self):
    input_shape = (self.height, self.width, self.depth)

    # Build the model and compile it.
    self.model = Sequential()
  
    self.model.add(Conv2D(32, (11, 11), input_shape=input_shape, padding="same"))
    self.model.add(BatchNormalization())
    self.model.add(Activation("relu"))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
  
    self.model.add(Conv2D(64, (7, 7), padding="same"))
    self.model.add(BatchNormalization())
    self.model.add(Activation("relu"))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Conv2D(128, (5, 5), padding="same"))
    self.model.add(BatchNormalization())
    self.model.add(Activation("relu"))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
  
    self.model.add(Conv2D(256, (3, 3), padding="same"))
    self.model.add(BatchNormalization())
    self.model.add(Activation("relu"))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))

    self.model.add(Flatten())
    self.model.add(Dense(512))
    self.model.add(BatchNormalization())
    self.model.add(Activation("relu"))
    self.model.add(Dropout(0.5))

    self.model.add(Dense(self.classes, activation="softmax"))

    self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

  def summary(self):
    print()
    print()
    print("CNN")
    print("--------------------")
    self.model.summary()

  def train(self, x_train, y_train, epochs, batch_size):
    history = self.model.fit(x_train, y_train, validation_split=0.25, epochs=epochs, batch_size=batch_size)

    if not os.path.isdir("./output"):
      os.mkdir("./output")

    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("CNN accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training set", "Validation set"], loc="upper left")
    plt.savefig("./output/cnn-fit-accuracy")
    plt.close()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("CNN loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training set", "Validation set"], loc="upper left")
    plt.savefig("./output/cnn-fit-loss")
    plt.close()

  def save(self):
    if not os.path.isdir("./output"):
      os.mkdir("./output")
    self.model.save("./output/cnn.h5")