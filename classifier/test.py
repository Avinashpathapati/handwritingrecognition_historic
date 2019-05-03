# Test module
# Module to test the classifier modules.

from utility import load_data, plot
from augmentation import rotate, translate, scale, augment
from cnn import build_cnn


data = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
data = augment(data)

#for image, label in zip(data["images"], data["labels"]):
  #plot(image, label)


#cnn = build_cnn()
#print(cnn.summary())

