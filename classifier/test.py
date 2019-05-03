# Test module
# Module to test the classifier modules.

from utility import load_data, plot, rotate, translate, scale


data = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")

plot(data["images"][0], "before")
plot(scale(data["images"][0]), "after")


# for image, label in zip(data["images"], data["labels"]):
  #plot(image, label)
