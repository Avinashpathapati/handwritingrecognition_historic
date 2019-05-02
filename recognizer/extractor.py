from recognizer.utility import load_single_image,plot,plot_matplotlib
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

		num, im, mask, rect = cv.floodFill(im, mask, seed, (1), (0,) * 3, (0,) * 3, floodflags)

		return mask

	def testing_start(self):
		image_path = '../data/test/'
		image_name = '0_test.jpg'



		img = self.load_image(image_path,image_name,load_greyscale=True)

		output_img = self.area_closing(img)
		output_img = self.area_opening(output_img)


		output_img = thresholded_binarisation(output_img,25) #Hypeparameter is threshold

		output_img = self.get_mask_of_centre(output_img)


		plot_matplotlib(output_img)


