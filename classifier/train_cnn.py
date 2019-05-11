# Train CNN module
# Author: Andreas Pentaliotis
# Module to implement training of cnn on the character data.

import numpy as np

from utility import load_data, randomize
from preprocessing import preprocess_training
from splitting import split
from cnn import CNN


images, labels = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
classes = len(np.unique(labels))

images, labels = preprocess_training(images, labels)
images, labels = randomize(images, labels)
x_train, x_test, y_train, y_test = split(images, labels)

# Delete the images and labels arrays to free memory.
del images
del labels

cnn = CNN(x_train.shape[1], x_train.shape[2], x_train.shape[3], classes)
cnn.summary()

cnn.train(x_train, y_train, epochs=20, batch_size=32, augment_data=False)
cnn.test(x_test, y_test)
cnn.save()
