import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from recognizer.utility import random_color



class CharacterSegmentationMeanShift(object):

	def __init__(self,img):
		self.img = abs(255 - img)

		self.img_shape = img.shape


	def invert_image(self,img):
		return abs(255-img)

	def plot_characters(self):
		flat_image = np.reshape(self.img, [-1, 1])

		# Estimate bandwidth
		bandwidth2 = estimate_bandwidth(flat_image,
										quantile=.3, n_samples=1000)

		ms = MeanShift(bandwidth2, bin_seeding=False)
		ms.fit(flat_image)
		labels = ms.labels_


		# Plot image vs segmented image
		plt.figure(2)
		plt.subplot(2, 1, 1)
		plt.imshow(self.invert_image(self.img))
		plt.axis('off')
		plt.subplot(2, 1, 2)

		plt.imshow(self.get_colored_characters(labels))
		plt.axis('off')

		plt.show()


	def get_colored_characters(self,labels):
		preview = np.ones((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)

		labels2d = np.reshape(labels, self.img_shape)

		unique_labels = np.unique(labels)

		for ul in unique_labels:
			color = random_color()
			preview[labels2d == ul]=color

		# for i in range(self.img_shape[0]):
		# 	for j in range(self.img_shape[1]):
		#
		# 		preview[i,j]=(0,0,0)

		return preview




