import numpy as np
import cv2 as cv

'''
For middle part extraction
'''


class ExtractorByOpening():
    def __init__(self, kernel_size):
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    def area_closing(self, img):
        closing = cv.morphologyEx(img, cv.MORPH_CLOSE, self.kernel)
        return closing

    def area_opening(self, img):
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, self.kernel)
        return opening

    def get_biggest_component(self, image):
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        mask = np.zeros(output.shape)
        mask[output == max_label] = 255
        mask = mask.astype(np.uint8)

        return mask

    def otsu_binarisation(self, image):
        image_otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        return image_otsu

    def extract_text(self, image):
        mask = self.area_closing(image)
        mask = self.area_opening(mask)
        mask = self.otsu_binarisation(mask)
        mask = self.get_biggest_component(mask)
        image = cv.bitwise_and(image, image, mask=mask)
        return image, mask



