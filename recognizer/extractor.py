from recognizer.utility import load_single_image, plot_opencv, plot_matplotlib,plot_histogram
from recognizer.preprocess import binarize,thresholded_binarisation,niblack_and_otsu_binarisation
import numpy as np
import cv2 as cv

'''
For middle part extraction
'''
class ExtractorByOpening():
	def __init__(self,kernel_size):
		self.kernel = np.ones((kernel_size,kernel_size),np.uint8)

	def load_image(self,image_path,image_name,load_greyscale=False):
		return load_single_image(image_path,image_name,load_greyscale=load_greyscale)

	def area_closing(self,img):
		closing = cv.morphologyEx(img, cv.MORPH_CLOSE, self.kernel)
		return closing

	def area_opening(self,img):
		opening = cv.morphologyEx(img, cv.MORPH_OPEN, self.kernel)
		return opening

	def filter_cc(self,img):
		output = cv.connectedComponentsWithStats(image=img)
		print(output[0])
		print(output[1])
		print(output[2])

	def save_dialated_image(self,image_path,image_name):
		pass

	def find_nearest_white(self, image, target):
		nonzero = np.argwhere(image == 255)
		distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
		nearest_index = np.argmin(distances)
		return nonzero[nearest_index]

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

	def adjust_gamma(self,image, gamma=1.0):

		invGamma = 1.0 / gamma
		table = np.array([((i / 255.0) ** invGamma) * 255
						  for i in np.arange(0, 256)]).astype("uint8")

		return cv.LUT(image, table)


	def otsu_binarisation(self,image):
		image_otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
		return image_otsu


	def extract_text(self, image):
		mask = self.area_closing(image)
		mask = self.area_opening(mask)
		mask = self.otsu_binarisation(mask)
		mask = self.get_biggest_component(mask)
		image = cv.bitwise_and(image, image, mask = mask)
		return image, mask

	def testing_start(self):
		image_path = '../data/test/imgs/'
		#image_path = '../data/test/tmp/'
		#image_name = 'another_sample.jpg'
		#image_name = 'character_test.png'
		image_name = '8_test.jpg'
		#image_name = 'small.png'
		#image_name = 'line2.png'
		#image_name = '2634.png'

		image = self.load_image(image_path, image_name, load_greyscale=True)
		#image = self.area_closing(image)
		#image = self.area_opening(image)
		#image = thresholded_binarisation(image, 25)
		#image = self.get_biggest_component(image)
		#image = self.get_mask_of_centre(image)
		
		image = self.extract_text(image)
		return image



