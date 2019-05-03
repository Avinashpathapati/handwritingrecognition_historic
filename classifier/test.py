# Test module
# Module to test the classifier modules.

from utility import load_data, plot
from augmentation import rotate, translate, scale, augment
from cnn import build_cnn


images, labels = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
images, labels = augment(images, labels)

#for image, label in zip(images, labels):
  #plot(image, label)

#cnn = build_cnn()
#print(cnn.summary())

