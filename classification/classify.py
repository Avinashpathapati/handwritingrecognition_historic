from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class CNN:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def make_predictions(self, images, generator=None):
        if generator is None:
            predictions = self.model.predict(images)
        else:
            # Make the predictions by performing test time data augmentation using the given generator. For
            # this to make sense the model should have been trained with the same generator.
            predictions = []
            for _ in range(20):
                current_predictions = self.model.predict_generator(generator.flow(
                    images, batch_size=1, shuffle=False
                ), steps=images.shape[0])
                predictions.append(current_predictions)
            predictions = np.mean(predictions, axis=0)

        return predictions
