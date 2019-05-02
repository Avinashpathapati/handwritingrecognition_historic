from recognizer.utility import load_single_image, plot_opencv, plot_matplotlib
from recognizer.preprocess import binarize,thresholded_binarisation
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

	def get_mask_of_centre(self,im):
		h, w = im.shape
		seed = (int(w / 2), (int)(h / 2))

		mask = np.zeros((h + 2, w + 2), np.uint8)


		floodflags = 4
		floodflags |= cv.FLOODFILL_MASK_ONLY
		floodflags |= (255 << 8)

		num, im, mask, rect = cv.floodFill(im, mask, seed, 1, 0, 0, floodflags)
		mask = cv.resize(mask, (w, h))
		return mask

	def get_center(self, image):
		mask = self.area_closing(image)
		mask = self.area_opening(mask)
		mask = thresholded_binarisation(mask,25) #Hypeparameter is threshold
		mask = self.get_mask_of_centre(mask)
		return cv.bitwise_and(image, image, mask = mask)

	def testing_start(self):
		image_path = '/home/anpenta/Desktop/handwriting-recognizer/data/image-data/'
		image_name = 'P423-1-Fg002-R-C01-R01-fused.jpg'

		image = self.load_image(image_path,image_name,load_greyscale=True)
		image = self.get_center(image)
		
		plot_matplotlib(image)


