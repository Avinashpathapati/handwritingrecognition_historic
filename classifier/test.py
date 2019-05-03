# Test module
# Module to test the classifier modules.

from utility import load_data, plot


data = load_data("/home/anpenta/Desktop/character-classifier/data/monkbrill-jpg/monkbrill2")
for image, label in zip(data["images"], data["labels"]):
  plot(image, label)
