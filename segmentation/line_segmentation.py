
import cv2 as cv
import numpy as np

from recognizer.utility import plot_opencv,save_opencv
from external_code.horizontal_profiling import get_valleys

from external_code.astar import Astar
from utility.utility import timeit
from external_code.jps import Jps
from segmentation.graph_based import GraphLSManager
from segmentation.water_flow import WaterFlow
import matplotlib.pyplot as plt
from external_code.horizontal_profiling import projection_analysis

class LineSegmentation():
	def __init__(self):
		pass


	def tech_graph_based(self,img):
		method = GraphLSManager(img)

		new_img,lines_images = method.start()

		return new_img,lines_images

	def tech_water_flow(self,img):
		method = WaterFlow(img)

		new_img = method.run()

		return new_img

	def segment_lines(self,img):
		img,line_images = self.tech_graph_based(img)
		return img,line_images

	@timeit
	def test_segmentation(self,img):
		img = self.tech_graph_based(img)
		plot_opencv(img)
		return

