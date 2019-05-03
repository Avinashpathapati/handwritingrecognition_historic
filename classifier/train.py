# Train module
# Module to implement training of a model on the character data.

import numpy as np

from utility import load_data, pad, plot
from augmentation import augment
from cnn import build_cnn


# Load the data, augment them, shuffle them with a seed and get the number of classes.
data = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
data = augment(data)
data = data.sample(frac=1, random_state=1).reset_index(drop=True)
classes = data["labels"].nunique()

# Pad all the images with white pixels to maximum height and maximum width.
max_width = np.amax(np.unique([x.shape[1] for x in data["images"]]))
max_height = np.amax(np.unique([x.shape[0] for x in data["images"]]))
data["images"] = [pad(x, max_width, max_height) for x in data["images"]]

#for image, label in zip(data["images"], data["labels"]):
  #plot(image, label)

#cnn = build_cnn(None, None, 1, classes)
