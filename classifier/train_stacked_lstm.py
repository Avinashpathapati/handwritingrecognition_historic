# Train stacked LSTM module
# Author: Andreas Pentaliotis
# Module to implement training of stacked LSTM on the character data.

import numpy as np

from utility import load_data
from preprocessing import preprocess_training
from splitting import split
from augmentation import augment
from stacked_lstm import StackedLSTM


images, labels = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
classes = len(np.unique(labels))

images, labels = augment(images, labels)
images, labels = preprocess_training(images, labels)
x_train, x_test, y_train, y_test = split(images, labels, "stacked_lstm")

# Delete the images and labels arrays to free memory.
del images
del labels

stacked_lstm = StackedLSTM(x_train.shape[1], x_train.shape[2], classes)
stacked_lstm.summary()

stacked_lstm.train(x_train, y_train, epochs=1, batch_size=32)
stacked_lstm.test(x_test, y_test)
stacked_lstm.save()
