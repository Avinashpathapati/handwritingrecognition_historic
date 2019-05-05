# Train module
# Author: Andreas Pentaliotis
# Module to implement training of a model on the character data.

import numpy as np
from utility import load_data
from preprocessing import preprocess
from augmentation import augment
from cnn import CNN


images, labels = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
classes = len(np.unique(labels))

images, labels = augment(images, labels)
x_train, x_test, y_train, y_test = preprocess(images, labels)

# Delete the original lists to free memory.
del images[:]
del labels[:]

cnn = CNN(x_train.shape[1], x_train.shape[2], x_train.shape[3], classes)
cnn.summary()

cnn.train(x_train, y_train, epochs=500, batch_size=32)
cnn.save()
