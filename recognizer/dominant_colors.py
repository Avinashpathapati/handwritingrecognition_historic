

from sklearn.neighbors.kde import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from recognizer.utility import plot_histogram,plot_opencv
from sklearn.cluster import KMeans

class DominantColors(object):
	def __init__(self,num_clusters=3):
		self.clusters=3

	'''
	image should be a grey scale image
	'''
	def find_dominant_colors(self,image):
		#plot_opencv(image)
		img_H = image.shape[0]
		img_W= image.shape[1]

		flat_imgs = image.reshape(img_H*img_W)
		#plot_histogram(flat_imgs, ylim=[0, 7000])
		#print(len(flat_imgs))

		black_point_indexes = np.where(flat_imgs == 0)
		flat_imgs=np.delete(flat_imgs,black_point_indexes)

		plot_histogram(flat_imgs, ylim=[0, 7000])

		# feature_vector = np.ones((len(flat_imgs),2))
		#
		# for i in range(len(flat_imgs)):
		# 	feature_vector[i][0]=flat_imgs[i]
		#
		# kmeans = KMeans(n_clusters=self.clusters)
		# kmeans.fit(feature_vector)
		#
		# self.distinct_colors= kmeans.cluster_centers_
		#
		# self.labels = kmeans.labels_
		#
		# print(len(self.labels),len(feature_vector),len(flat_imgs))
		#
		# final_img = np.zeros((img_H,img_W))
		#
		# for j in range(img_W):
		# 	for i in range(img_H):
		# 		final_img[i][j] = self.distinct_colors[self.labels[((i*(img_W-1))+j+1)]][0]
		#
		# plot_opencv(final_img)
		return


	def plot(self,img):
		pass
