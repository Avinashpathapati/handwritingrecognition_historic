# Stacked LSTM module
# Author: Andreas Pentaliotis
# Stacked long short-term memory neural network implementation.

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import BatchNormalization
from keras import backend as K
K.set_image_dim_ordering("tf")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


class StackedLSTM():
  def __init__(self, height, width, classes):
    self.height = height
    self.width = width
    self.classes = classes
    self.__build()

  def __build(self):
    input_shape = (self.height, self.width)

    # Build the model and compile it.
    self.model = Sequential()
  
    self.model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    self.model.add(LSTM(128))

    self.model.add(Dense(512))
    self.model.add(BatchNormalization())
    self.model.add(Activation("relu"))
    self.model.add(Dropout(0.5))    

    self.model.add(Dense(self.classes, activation="softmax"))

    self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

  def summary(self):
    print()
    print()
    print("Stacked LSTM")
    print("--------------------")
    self.model.summary()

  def train(self, x_train, y_train, epochs, batch_size):
    fitting = self.model.fit(x_train, y_train, validation_split=0.25, epochs=epochs, batch_size=batch_size)

    if not os.path.isdir("./output"):
      os.mkdir("./output")

    plt.plot(fitting.history["acc"])
    plt.plot(fitting.history["val_acc"])
    plt.title("Stacked LSTM accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training set", "Validation set"], loc="upper left")
    plt.savefig("./output/stacked-lstm-fitting-accuracy")
    plt.close()

    plt.plot(fitting.history["loss"])
    plt.plot(fitting.history["val_loss"])
    plt.title("Stacked LSTM loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training set", "Validation set"], loc="upper left")
    plt.savefig("./output/stacked-lstm-fitting-loss")
    plt.close()

  def test(self, x_test, y_test):
    evaluation = self.model.evaluate(x_test, y_test)

    if not os.path.isdir("./output"):
      os.mkdir("./output")

    with open("./output/stacked-lstm-evaluation.txt", "w+") as output_file:
      output_file.write("Model's loss on unseen data: " + str(evaluation[0]))
      output_file.write("\nModel's accuracy on unseen data: " + str(evaluation[1]))

  def save(self):
    if not os.path.isdir("./output"):
      os.mkdir("./output")
    self.model.save("./output/stacked-lstm.h5")