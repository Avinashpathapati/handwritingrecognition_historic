import cv2 as cv
import numpy as np
from recognizer.utility import random_color,show_color_histogram,save_opencv
from recognizer.utility import plot_opencv
import copy

from sklearn.linear_model import LinearRegression

'''
Node of the graph representing connected component
Note x represent row and y represent columns
'''
class GraphNode(object):
	def __init__(self,label,gc_x,gc_y,area,topx,topy,h,w):
		self.label=label
		self.gc_x = gc_x
		self.gc_y = gc_y
		self.color_assigned= None
		self.area = area

		#self.gc_x = topx + int(round(h/2))
		#self.gc_y = topy + int(round(w/2))

	def __eq__(self, other):
		return ((self.gc_x == other.gc_x) and (self.gc_y == other.gc_y) and (self.label == other.label))

	def check_on_same_line(self,other):
		if abs(self.gc_x-other.gc_x) > 20: #PARAMETER TO TUNE
			return False
		dist = (((self.gc_x - other.gc_x) ** 2) + ((self.gc_y - other.gc_y) **2))**0.5
		try:
			angle = np.tanh(abs(self.gc_x - other.gc_x) / abs(self.gc_y - other.gc_y)) * (180/np.pi)
		except :
			angle =90

		if 0<=angle<=15:
			return True
		elif dist < 10: #PARAMETER TO TUNE
			return True

		return  False

	def __str__(self):
		return str(self.label)+','+str(self.gc_x)+','+str(self.gc_y)





class GraphBasedLS():
	def __init__(self,img):
		self.img = img
		self.all_cc = []
		#self.preprocess()

	def preprocess(self):
		kernel = np.ones((3, 3), np.uint8)
		img = cv.erode(self.img, kernel, iterations=1)
		#img = cv.morphologyEx(self.img, cv.MORPH_OPEN,kernel)
		self.img = img
		return


	def get_centroids(self):
		image = 255 - self.img  # Represent white as text and background as black
		nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
		self.output =  output
		#print(np.unique(self.output))


		left_cols = stats[:,0]
		top_rows = stats[:,1]
		widths = stats[:,2]
		heights = stats[:,3]
		areas = stats[:, 4]


		new_image = np.zeros(image.shape)

		print('TOTAL COMPONENTS:',nb_components)
		filtered_components= 0
		for i in range(1, nb_components):
			area_of_component = areas[i]
			#print(area_of_component)
			if area_of_component > 0: #PARAMETER HERE
				centroid_y = int(round(centroids[i,0]))
				centroid_x = int(round(centroids[i,1]))
				#print(centroid_x,centroid_y)
				new_image[centroid_x,centroid_y]=255
				filtered_components = filtered_components +1
				#self.img[int(round(centroid_y)),int(round(centroid_x))] = 255
				new_node = GraphNode(i,centroid_x,centroid_y,area_of_component,top_rows[i],left_cols[i],heights[i],widths[i])
				self.all_cc.append(new_node)

		print('FILTERED COMPONENTS:', filtered_components)

		self.cc_num = filtered_components #Number of components

		#new_image = new_image -255
		#plot_opencv(new_image)
		return new_image


	def get_lines(self):
		ccs = self.all_cc
		lines=[]

		ccs = sorted(ccs, key= lambda x: x.gc_x)
		while ccs:
			#print(len(ccs))
			line = []
			start = ccs[0]
			line.append(start)
			ccs.remove(start)
			for cc in ccs:
				if start.check_on_same_line(cc):
					line.append(cc)
					ccs.remove(cc)

			lines.append(line)

		return lines


	def run(self):
		img = self.get_centroids()
		lines = self.get_lines()
		n_lines = len(lines)
		print(n_lines)
		ntimes=1
		while ntimes > 0:
			lines = self.cluster_lines(lines, 11-ntimes)
			n_lines = len(lines)
			print(n_lines)
			ntimes=ntimes-1


		# lines= self.cluster_lines(lines,10)
		# print(len(lines))
		# lines = self.cluster_lines(lines,7)
		# print(len(lines))
		#
		# lines = self.cluster_lines(lines, 8)
		# print(len(lines))
		#
		# n_lines = len(lines)
		# while n_lines > 30:
		# 	lines = self.cluster_lines(lines, 10)
		# 	n_lines = len(lines)
		# 	print(len(lines))

		#img = self.get_one_line(lines[57])
		self.give_color_to_lines(lines)

		# img = self.draw_colored_centroids()
		# save_opencv(img, '../data/test/', '25.png')

		img = self.draw_colored_lines()
		# save_opencv(img, '../data/test/', '23.png')

		# self.adjust_colors()
		# img = self.draw_colored_lines()
		# save_opencv(img, '../data/test/', '24.png')

		# img,lines=self.adjust_colors_regression(lines)
		#
		# save_opencv(img, '../data/test/', '26.png')
		#
		# img = self.filter_lines_with_more_components(lines)
		# save_opencv(img, '../data/test/', '27.png')

		return img


	def filter_lines_with_more_components(self,lines):
		reclassify = []
		correct = []

		for line in lines:
			if len(line) > 15:
				correct.append(line)
			else:
				reclassify.append(line)

		return self.draw_colored_lines_from_given_lines(correct)

	def adjust_colors_regression(self,lines):
		reclassify = []
		correct = []

		models = []

		for line in lines:
			if len(line)>15:
				correct.append(line)
			else:
				reclassify.append(line)


		for c_line in correct:
			num = len(c_line)
			X = np.ones((num,1))
			Y= np.ones((num,1))
			for i in range(num):
				cc = c_line[i]
				X[i,0] = cc.gc_y
				Y[i,0]= cc.gc_x
			model = LinearRegression().fit(X, Y)

			models.append(model)


		for rc_line in reclassify:
			for cc in rc_line:
				x= cc.gc_y
				y=cc.gc_x

				X = np.ones((1,1))
				X[0,0]=x
				results = []
				for model in models:
					result = model.predict(X)
					result = abs(y-result[0])
					results.append(result)

				if results:
					min_index = results.index(min(results))

					new_color = correct[min_index][0].color_assigned
					cc.color_assigned=new_color
					correct[min_index].append(cc)

		return self.draw_colored_lines_from_given_lines(correct),correct










	def cluster_lines(self,lines,x_th):
		new_lines=[]
		average_x = []

		total_cc_added=0
		total_cc_provided=0

		i =0
		for line in lines:
			nb_cc_l = len(line)
			total_cc_provided = total_cc_provided+nb_cc_l
			sum_x=0
			for cc in line:
				sum_x = sum_x + cc.gc_x

			avg_cc_x = (i,round(sum_x/nb_cc_l))
			i=i+1
			average_x.append(avg_cc_x)


		sorted_by_x = sorted(average_x,key=lambda x:x[1])

		print(len(sorted_by_x))

		flag = True
		last_line_v =  sorted_by_x[0]
		nline=[]
		for i in range(1,len(sorted_by_x)):
			if flag:
				nline=lines[last_line_v[0]]
				total_cc_added = total_cc_added +len(nline)
				flag = False

			current_line_v = sorted_by_x[i]
			diff = abs(current_line_v[1] - last_line_v[1])

			if i>2:
				diff2 = abs(last_line_v[1] - last_last_line_v[1])
				#diff3 = abs(last_last_line_v[1]-last_last_last_line_v[1])

				#diff = (diff+diff2+diff3)/3
				diff = (diff + diff2)/2


			if diff > x_th:
				flag = True
				new_lines.append(nline)
			else:
				total_cc_added = total_cc_added + len(lines[current_line_v[0]])
				nline.extend(lines[current_line_v[0]])

			if i >=2:
				#last_last_last_line_v = sorted_by_x[i-2]
				last_last_line_v = last_line_v


			last_line_v = current_line_v

		new_lines.append(lines[last_line_v[0]])
		#print(new_lines)
		print('TOTAL',total_cc_added,total_cc_provided)
		return new_lines





	def draw_colored_centroids(self):
		preview = np.ones((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)*200
		#preview = (255,255,255)
		for cc in self.all_cc:
			preview[self.output == cc.label] = (150,150,150)

		for cc in self.all_cc:
			preview[cc.gc_x, cc.gc_y] = (0,0,0)#cc.color_assigned

		return preview

	def draw_colored_lines(self):
		preview = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
		for cc in self.all_cc:
			if cc.color_assigned:
				preview[self.output == cc.label]=cc.color_assigned

		return preview


	def draw_colored_lines_from_given_lines(self,lines):
		preview = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
		for line in lines:
			for cc in line:
				preview[self.output == cc.label]=cc.color_assigned

		return preview


	def adjust_colors(self):
		#ccs_copy = copy.deepcopy(self.all_cc)
		self.all_cc = sorted(self.all_cc,key=lambda x: x.area,reverse=True)


		for i in range(len(self.all_cc)):
			cc = self.all_cc[i]
			color_choices=[(cc.color_assigned,cc.area)]
			for j in range(len(self.all_cc)):
				if j!=i:
					cc2=self.all_cc[j]
					#dist = (((cc.gc_x - cc2.gc_x) ** 2) + ((cc.gc_y - cc2.gc_y) **2))**0.5

					distx = abs(cc.gc_x - cc2.gc_x)
					disty = abs(cc.gc_y - cc2.gc_y)

					if distx<10 and disty < 1000: #and cc2.area>1000:
					#if dist < 50 and cc.area
						color_choices.append((cc2.color_assigned,cc2.area))

			most_common_color = self.get_most_common_color(color_choices)

			cc.color_assigned=most_common_color
			#ccs_copy[i].color_assigned = most_common_color


		#self.all_cc = ccs_copy




	def get_most_common_color(self,color_choices):
		colors={}

		for val in color_choices:
			color, area = val[0],val[1]
			if color in colors:
				colors[color] = colors[color]+area
			else:
				colors[color]=area

		#print(colors)
		max_freq=0
		max_color=(0,0,0)
		for k,v in colors.items():
			if v > max_freq:
				max_color=k

		return max_color



	def give_color_to_lines(self,lines):
		preview = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)

		for line in lines:
			color = random_color()
			for cc in line:
				#preview[self.output == cc.label] = color
				cc.color_assigned=color

		#show_color_histogram(preview)
		#self.draw_colored_centroids()
		return



	def get_one_line(self,line):
		labels = []
		new_image = np.zeros(self.img.shape)
		for cc in line:
			labels.append(cc.label)
			new_image[self.output == cc.label] =255
		print(labels)


		return new_image


