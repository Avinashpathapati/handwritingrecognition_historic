import cv2 as cv
import numpy as np
from recognizer.utility import random_color,show_color_histogram,save_opencv
from recognizer.utility import plot_opencv
import copy

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

from external_code.horizontal_profiling import get_valleys,projection_analysis
import random

from skimage.transform import probabilistic_hough_line
from segmentation.water_flow import WaterFlow

'''
Node of the graph representing connected component
Note x represent row and y represent columns
'''

import sys

np.set_printoptions(threshold=sys.maxsize)

class GravityCenter(object):
	def __init__(self,ax,ay,x,y):
	#def __int__(self,ax,ay,x,y):
		self.ax=ax
		self.ay= ay
		self.x=x
		self.y=y

class GraphNode(object):
	def __init__(self,label,gc_x,gc_y,area,topx,topy,h,w,cc_area_for_no_impact):
		self.label=label
		self.gc_x = gc_x
		self.gc_y = gc_y
		self.color_assigned= None
		self.area = area

		self.topx = topx
		self.topy = topy
		self.h=h
		self.w = w
		self.cc_area_for_no_impact = cc_area_for_no_impact

		#self.gc_x = topx + int(round(h/2))
		#self.gc_y = topy + int(round(w/2))

	def set_center(self,centers):
		self.centers = centers

		if centers:
			self.update_g_x_y()

	def update_g_x_y(self):
		gc_x=0
		gc_y=0

		for center in self.centers:
			gc_x = gc_x + center.ax
			gc_y = gc_y + center.ay

		self.gc_x = int(round(gc_x/len(self.centers)))
		self.gc_y = int(round(gc_y/len(self.centers)))


	def __eq__(self, other):
		return ((self.gc_x == other.gc_x) and (self.gc_y == other.gc_y) and (self.label == other.label))

	def check_on_same_line(self,other):
		if other.area < self.cc_area_for_no_impact:#No considering smaller components for classifying bigger
			return False
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


	def check_on_same_line2(self,other):
		if other.area < self.cc_area_for_no_impact:#No considering smaller components for classifying bigger
			return False
		if abs(self.gc_x-other.gc_x) > 20: #PARAMETER TO TUNE
			return False


		sum_angle=0
		sum_dist=0
		#for center1 in self.centers:
		for center2 in other.centers:
			# dist = (((center1.ax - center2.ax) ** 2) + ((center1.ay - center2.ay) ** 2)) ** 0.5
			# try:
			# 	angle = np.tanh(abs(center1.ax - center2.ax) / abs(center1.ay - center2.ay)) * (180/np.pi)
			# except :
			# 	angle =90

			dist = (((self.gc_x - center2.ax) ** 2) + ((self.gc_y - center2.ay) ** 2)) ** 0.5
			try:
				angle = np.tanh(abs(self.gc_x - center2.ax) / abs(self.gc_y - center2.ay)) * (180/np.pi)
			except :
				angle =90

			sum_angle = sum_angle + angle
			sum_dist = sum_dist + dist

		# angle = sum_angle/(len(self.centers)* len(other.centers))
		# dist = sum_dist/(len(self.centers)* len(other.centers))

		angle = sum_angle/len(other.centers)
		dist = sum_dist/len(other.centers)



		if 0<=angle<=15:
			return True
		elif dist < 10: #PARAMETER TO TUNE
			return True

		return  False

	def __str__(self):
		return str(self.label)+','+str(self.gc_x)+','+str(self.gc_y)




class GraphLSManager():
	def __init__(self,img):
		self.img=img

		self.output = np.ones(self.img.shape)


	def start(self):

		method = GraphBasedLS(self.img)
		lines,line_images = method.run()

		# lines = self.post_process(lines,line_images)
		# img = self.draw_colored_lines_from_given_lines(lines)

		img = method.draw_colored_lines_from_given_lines(lines)
		img= method.draw_blocks(img,lines)
		img= method.draw_peak_lines(img)

		#save_opencv(img,'../data/test/','1236.png')
		return img

	def draw_colored_lines_from_given_lines(self,lines):
		preview = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
		for line in lines:
			for cc in line:
				preview[self.output == cc.label]=cc.color_assigned

		return preview

	def post_process(self,lines,line_images):
		# img = abs(255 - line_images[4])
		# img  = img.astype(np.uint8)
		# method = GraphBasedLS(img)
		# lines2= method.run(with_line_images=False)
		# img=method.draw_colored_lines_from_given_lines(lines2)
		# plot_opencv(img)
		#
		# return lines

		num_peaks = []
		average_num_pixels=[]
		new_lines = []
		for i in range(len(lines)):
			img= line_images[i]

			#peaks,hist=projection_analysis(img/255,return_hist=True)

			#print(hist)

			#num_peaks.append(len(peaks))

			#average_num_pixels.append(np.average(hist))

			#print('Peak,Average',len(peaks),np.average(hist))

			img = abs(255 - img)
			img  = img.astype(np.uint8)

			# if i < 4:
			# 	plot_opencv(img)
			method = GraphBasedLS(img,cc_area_for_no_impact=0)
			lines2,line_images2 = method.run()

			#print(line_images2)

			for j in range(len(lines2)):
				new_lines.append((lines2[j],line_images2[j]))



		# mean_num_peaks = np.mean(num_peaks)
		# for i in range(len(num_peaks)):
		# 	if num_peaks[i] > mean_num_peaks:
		# 		# total_line_area = 0
		# 		# for cc in lines[i]:
		# 		# 	total_line_area = total_line_area+cc.area
		#
		# 		if average_num_pixels[i] >100:
		# 			print('Redoing for line', i)
		# 			# img = abs(255-line_images[i])
		# 			# img  = img.astype(np.uint8)
		# 			# method = GraphBasedLS(img)
		# 			# lines2= method.run(with_line_images=False)
		# 			# # img=method.draw_colored_lines_from_given_lines(lines2)
		# 			# # plot_opencv(img)
		# 			# for l in lines2:
		# 			# 	new_lines.append(l)
		# 	else:
		# 		new_lines.append(lines[i])


		#return new_lines

		## RECOLOR and RELABEL
		cc_num=0
		for i in range(len(new_lines)):
			line = new_lines[i][0]
			line_image = new_lines[i][1]
			color = random_color()
			for cc in line:
				cc.label = cc_num
				cc.color_assigned = color
				self.output[line_image == 255]= cc_num

				cc_num = cc_num + 1

		print(np.unique(self.output))
		return lines


class GraphBasedLS():
	def __init__(self,img,cc_area_for_no_impact=100):
		self.img = img
		self.all_cc = []

		self.cc_area_for_no_impact = cc_area_for_no_impact #PARAMETER
		self.peaks=[]
		#self.preprocess()

	def preprocess(self):
		kernel = np.ones((3, 3), np.uint8)
		img = cv.erode(self.img, kernel, iterations=1)
		#img = cv.morphologyEx(self.img, cv.MORPH_OPEN,kernel)
		self.img = img
		return


	def get_centroids(self):
		image = 255 - self.img  # Represent white as text and background as black
		#plot_opencv(image)
		#save_opencv(self.img, '../data/test/tmp/', '2'+str(random.randint(1,1000))+'.png')
		nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
		#print('nb_components:',nb_components)
		self.output =  output
		#print('unique labels',np.unique(self.output))




		left_cols = stats[:,0]
		top_rows = stats[:,1]
		widths = stats[:,2]
		heights = stats[:,3]
		areas = stats[:, 4]

		AH = 0
		for i in range(1,nb_components):
			AH = AH + heights[i]

		self.AH = AH / nb_components-1 # nb_components is number of components
		self.AW = self.AH

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
				new_node = GraphNode(i,centroid_x,centroid_y,area_of_component,top_rows[i],left_cols[i],heights[i],widths[i],
									 self.cc_area_for_no_impact)

				cc_img = np.where(self.output == new_node.label)
				centers= self.get_gravity_centers_for_cc(cc_img)
				new_node.set_center(centers)

				self.all_cc.append(new_node)

		#print('FILTERED COMPONENTS:', filtered_components)

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
				if start.check_on_same_line2(cc):
					line.append(cc)
					ccs.remove(cc)

			lines.append(line)

		return lines


	def run(self,with_line_images=True):
		img = self.get_centroids()

		#lines = self.get_lines()
		lines = self.get_lines_water_flow()

		#lines = self.get_lines_hough()
		#print(len(lines))
		#lines = self.get_lines_probabilistic_hough()

		# ntimes=1
		# while ntimes > 0:
		# 	lines = self.cluster_lines(lines, 11-ntimes)
		# 	n_lines = len(lines)
		# 	#print(n_lines)
		# 	ntimes=ntimes-1


		self.give_color_to_lines(lines)



		# num_components = self.get_num_components_to_be_considerd_as_line(lines)
		# lines=self.adjust_colors_regression_4(lines,num_components)
		#

		lines = self.cluster_lines_h_range(lines)

		# num_components = self.get_num_components_to_be_considerd_as_line(lines)
		# lines=self.adjust_colors_regression_3(lines,num_components)



		# num_components = self.get_num_components_to_be_considerd_as_line(lines)
		# lines = self.adjust_colors_regression_2(lines, num_components)



		# save_opencv(img, '../data/test/', '26.png')
		#
		# img = self.filter_lines_with_more_components(lines)
		# save_opencv(img, '../data/test/', '27.png')

		# for line in lines:
		# 	pass

		#lines = self.cluster_lines_h_range(lines,10)
		#lines = self.cluster_lines_h_range(lines, 10)
		#lines = self.cluster_lines_peaks(lines,10)

		# img = self.draw_colored_lines_from_given_lines(lines)
		# plot_opencv(img)

		if with_line_images:
			line_images=[]
			for line in lines:
				line_images.append(self.get_one_line_inverted(line))

			return lines,line_images


		return lines





	def get_num_components_to_be_considerd_as_line(self,lines):
		nums=[]
		for line in lines:
			components_in_line = 0
			for cc in line:
				if cc.area > self.cc_area_for_no_impact:
					components_in_line = components_in_line + 1

			nums.append(components_in_line)


		mean = np.mean(nums)
		median = np.median(nums)
		std= np.std(nums)

		#return int(round(mean-std))
		#return int(round(mean))

		try:
			return int(round(median))
		except:
			return 0

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
			if len(line) > 15:
				correct.append(line)
			else:
				reclassify.append(line)


		for c_line in correct:
			num = len(c_line)
			#X = np.ones((num,1))
			#Y= np.ones((num,1))
			X = []
			Y = []
			for i in range(num):
				cc = c_line[i]
				if cc.area > self.cc_area_for_no_impact:#PARAMETER TO TUNE, same as
					X.append([cc.gc_y])
					Y.append([cc.gc_x])
					#X[i,0] = cc.gc_y
					#Y[i,0]= cc.gc_x

			X = np.array(X)
			Y = np.array(Y)
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

		return correct


	def adjust_colors_regression_2(self,lines,num_components):
		reclassified = {}
		correct = []

		models = []
		coeffs = []

		for line in lines:
			components_in_line=0
			for cc in line:
				if cc.area > self.cc_area_for_no_impact:
					components_in_line=components_in_line+1

			if components_in_line > num_components:
				correct.append(line)
			# else:
			# 	reclassify.append(line)


		for c_line in correct:
			num = len(c_line)
			#X = np.ones((num,1))
			#Y= np.ones((num,1))
			X = []
			Y = []
			for i in range(num):
				cc = c_line[i]
				if cc.area > self.cc_area_for_no_impact:#PARAMETER TO TUNE, same as
					#X.append([cc.gc_y,cc.area])
					X.append([cc.gc_y])
					Y.append([cc.gc_x])
					#X[i,0] = cc.gc_y
					#Y[i,0]= cc.gc_x

			if X:
				X = np.array(X).flatten()
				Y = np.array(Y).flatten()
				#model = LinearRegression().fit(X, Y)
				model = IsotonicRegression().fit(X, Y)
				#coeffs.append(model.coef_)
				models.append(model)


		print('No. of models',len(models))
		for trc_line in lines:
			for cc in trc_line:
				x= cc.gc_y
				y=cc.gc_x
				area=cc.area
				#X = np.ones((1,2))
				#X[0,0]=x
				#X=[[x,area]]
				X=[[x]]
				Y=[[y]]
				X=np.array(X).flatten()
				Y=np.array(Y).flatten()
				results = []

				######
				for model in models:
					result = model.predict(X)
					result = abs(y-result[0])
					#print(result)
					results.append(result)


				results = np.array(results)
				where_are_NaNs = np.isnan(results)
				results[where_are_NaNs] = float('inf')
				results = list(results)
				#######


				# for model in models:
				# 	result = model.score(X,Y)
				# 	results.append(result)

				#print(results)

				#########

				# if results:
				# 	results_copy = results
				# 	results_copy = sorted(results)
				# 	#print('all results',results_copy)
				# 	min_three = results_copy[:3]
				# 	#print('minimum 3',min_three)
				#
				# 	#min_index = results.index(min(results))
				#
				# 	min_index1 = results.index(min_three[0])
				# 	min_index2 = results.index(min_three[1])
				# 	min_index3 = results.index(min_three[2])
				# 	indexes = [min_index1,min_index2,min_index3]
				# 	coef1 = abs(coeffs[min_index1])
				# 	coef2 = abs(coeffs[min_index2])
				# 	coef3 = abs(coeffs[min_index3])
				# 	all_coeffs=[coef1,coef2,coef3]
				# 	#print('Coeffs O :',[coeffs[min_index1],coeffs[min_index2],coeffs[min_index3]])
				# 	#print('Coeffs',all_coeffs)
				# 	min_index = indexes[all_coeffs.index(min(all_coeffs))]

				###############

				if results:
					min_index = results.index(min(results))

					new_color = correct[min_index][0].color_assigned
					cc.color_assigned=new_color

					try:
						reclassified[min_index].append(cc)
					except:
						#reclassified.insert(min_index,[cc])
						#correct[min_index].append(cc)
						reclassified[min_index] = [cc]



		lines = []
		for key, val in reclassified.items():
			lines.append(val)
		return lines


	def get_mse_with_model(self,model,cc):
		mse=0
		num=0
		for r in range(cc.topx,cc.topx+cc.h,1):
			for c in range(cc.topy,cc.topy+cc.w,1):
				if self.img[r,c] == 0:
					X = [[c]]
					X = np.array(X)
					y = model.predict(X)[0]
					num=num+1
					mse=mse+(y**2)

		mse=mse/num

		return mse


	def cluster_lines_h_range(self,lines):
		new_lines=[]
		not_imp_lines = []
		hrange=[]

		total_cc_added = 0
		total_cc_provided = 0


		i = 0
		for line in lines:
			total_cc_provided = total_cc_provided+ len(line)
			nb_cc_l = 0
			top_x = 0
			bot_x = 0
			for cc in line:
				if cc.area > self.cc_area_for_no_impact:
					nb_cc_l = nb_cc_l + 1
					top_x = top_x + cc.topx
					bot_x = bot_x + cc.topx+cc.h
					# if cc.topx<top_x:
					# 	top_x=cc.topx
					# if cc.topx+cc.h > bot_x:
					# 	bot_x = cc.topx+cc.h




			if nb_cc_l:
				hrange.append([i,top_x/nb_cc_l,bot_x/nb_cc_l])
			else:
				not_imp_lines.append(line)

			i = i + 1

		sorted_by_x = sorted(hrange,key=lambda x:x[1])

		if len(sorted_by_x)==0:
			print('*******************************')
			self.give_color_to_lines(not_imp_lines)
			return not_imp_lines
		#print(sorted_by_x)
		#print(len(sorted_by_x))

		flag = True
		last_line_v =  sorted_by_x[0]
		nline=[]
		for i in range(1,len(sorted_by_x)):
			if flag:
				nline=lines[last_line_v[0]]
				total_cc_added = total_cc_added +len(nline)
				flag = False

			current_line_v = sorted_by_x[i]

			diff = current_line_v[1] - last_line_v[2] #top of current - bottom of last
			if diff > 0:
				flag = True
				new_lines.append(nline)

			else:
				total_cc_added = total_cc_added + len(lines[current_line_v[0]])
				nline.extend(lines[current_line_v[0]])


			last_line_v = current_line_v

		#print(len(sorted_by_x),len(new_lines))
		if flag:
			#print('If flag')
			new_lines.append(lines[last_line_v[0]])
			total_cc_added = total_cc_added + len(lines[last_line_v[0]])
		else:
			new_lines.append(nline)




		for l in not_imp_lines:
			new_lines.append(l)
			#print('In not imp')
			total_cc_added = total_cc_added + len(l)

		print('TOTAL h range', total_cc_added, total_cc_provided)


		self.give_color_to_lines(new_lines)
		return new_lines

	def cluster_lines_peaks(self,lines,x_th):
		new_lines=[]
		imp_lines = []
		not_imp_lines=[]

		average_x= []

		total_cc_added=0
		total_cc_provided=0

		i =0
		for line in lines:
			#nb_cc_l = len(line)
			total_cc_provided = total_cc_provided+ len(line)
			nb_cc_l=0
			for cc in line:
				if cc.area > self.cc_area_for_no_impact:
					nb_cc_l = nb_cc_l+1


			if nb_cc_l:
				imp_lines.append([i,line])
			else:
				not_imp_lines.append(line)

			i = i + 1

		for val in imp_lines:
			line =val[1]
			index=val[0]
			img = self.get_one_line_inverted(line)
			peaks, hist = projection_analysis(img / 255, return_hist=True)

			# print(hist)
			if len(hist)>0:
				max_index= np.argmax(hist)
				#average_x.append((index,np.average(peaks)))
				average_x.append((index, peaks[max_index]))



		sorted_by_x = sorted(average_x,key=lambda x:x[1])

		print(sorted_by_x)
		#print(len(sorted_by_x))

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
				#total_cc_added = total_cc_added + len(nline)
			else:
				total_cc_added = total_cc_added + len(lines[current_line_v[0]])
				nline.extend(lines[current_line_v[0]])

			if i >=2:
				#last_last_last_line_v = sorted_by_x[i-2]
				last_last_line_v = last_line_v


			last_line_v = current_line_v

		if flag:
			print('If flag')
			new_lines.append(lines[last_line_v[0]])
			total_cc_added = total_cc_added + len(lines[last_line_v[0]])
		else:
			new_lines.append(nline)

		for l in not_imp_lines:
			new_lines.append(l)
			total_cc_added = total_cc_added + len(l)

		print('TOTAL Peaks', total_cc_added, total_cc_provided)
		self.give_color_to_lines(new_lines)
		return new_lines




	def cluster_lines(self,lines,x_th):
		new_lines=[]
		average_x = []

		not_imp_lines=[]

		total_cc_added=0
		total_cc_provided=0

		i =0
		for line in lines:
			#nb_cc_l = len(line)
			total_cc_provided = total_cc_provided+ len(line)
			nb_cc_l=0
			sum_x=0
			for cc in line:
				if cc.area > self.cc_area_for_no_impact:
					sum_x = sum_x + cc.gc_x
					nb_cc_l = nb_cc_l+1




			if nb_cc_l:
				avg_cc_x = (i,round(sum_x/nb_cc_l))
				average_x.append(avg_cc_x)
			else:
				not_imp_lines.append(line)

			i = i + 1



		sorted_by_x = sorted(average_x,key=lambda x:x[1])
		#print(sorted_by_x)
		#print(len(sorted_by_x))

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

		if flag:
			print('If flag')
			new_lines.append(lines[last_line_v[0]])
			total_cc_added = total_cc_added + len(lines[last_line_v[0]])
		else:
			new_lines.append(nline)

		#new_lines=new_lines.extend(not_imp_lines)
		for l in not_imp_lines:
			new_lines.append(l)
			total_cc_added = total_cc_added + len(l)

		print('TOTAL 1st method', total_cc_added, total_cc_provided)
		self.give_color_to_lines(new_lines)
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
		#preview = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)

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
		new_image = np.ones(self.img.shape,dtype=np.uint8) * 255
		for cc in line:
			labels.append(cc.label)
			new_image[self.output == cc.label] = 0
		#print(labels)


		return new_image


	def get_one_line_inverted(self,line):
		new_image = np.ones(self.img.shape)
		for cc in line:
			new_image[self.output == cc.label] = 255
		#print(labels)


		return new_image


	def get_gravity_centers_for_cc(self,cps):
		block_width = int(self.AW)
		centers=[]
		xs = cps[0]
		ys = cps[1]
		b_x_start = min(xs)
		b_x_end = max(xs)
		block_height = b_x_end - b_x_start

		gy = min(ys)
		y_end = max(ys)
		while gy < y_end:
			b_y_start = gy
			gy = gy + block_width
			b_y_end = gy if gy < y_end else y_end
			block = self.img[b_x_start:b_x_end, b_y_start:b_y_end]


			cY, cX = self.get_gravity_center_of_block(block)

			#centers.append((b_x_start + cX,b_y_start + cY))
			centers.append(GravityCenter(b_x_start + cX,b_y_start + cY,cX,cY))
		return centers


	def get_gravity_center_of_block(self,block):
		# calculate moments of binary image
		M = cv.moments(block)

		if M["m00"] != 0:
			cY = int(M["m10"] / M["m00"])
			cX = int(M["m01"] / M["m00"])
		else:
			cX, cY = 0, 0
		#print(cX,cY)
		return cY,cX


	def draw_blocks(self,image,lines):
		for line in lines:
			for cc in line:
				for center in cc.centers:
					x_start =  center.ay-center.y
					x_end = center.ay+center.y
					y_start=center.ax - center.x
					y_end = center.ax+center.x

					#print(x_start,y_start,x_end,y_end)
					cv.line(image, (x_start, y_start), (x_end, y_start), (100, 100), 1)
					cv.line(image, (x_end, y_start), (x_end, y_end), (100, 100), 1)
					cv.line(image, (x_end, y_end), (x_start, y_end), (100, 100), 1)
					cv.line(image, (x_start, y_end), (x_start, y_start), (100, 100), 1)

		return image



	def adjust_colors_regression_3(self,lines,num_components):
		reclassified = {}
		correct = []


		models = []


		for line in lines:
			components_in_line=0
			for cc in line:
				if cc.area > self.cc_area_for_no_impact:
					components_in_line=components_in_line+1

			if components_in_line > num_components:
				correct.append(line)



		for c_line in correct:
			num = len(c_line)

			X = []
			Y = []
			for i in range(num):
				cc = c_line[i]
				if cc.area > self.cc_area_for_no_impact:#PARAMETER TO TUNE, same as
					for center in cc.centers:
						X.append([center.ay])
						Y.append([center.ax])

			if X:
				X = np.array(X).flatten()
				Y = np.array(Y).flatten()
				#model = LinearRegression().fit(X, Y)
				model = IsotonicRegression().fit(X, Y)
				models.append(model)


		print('No. of models',len(models))
		for trc_line in lines:
			for cc in trc_line:
				X=[]
				Y=[]
				for center in cc.centers:
					x= center.ay
					y=center.ax
					X.append(x)
					Y.append(y)


				X=np.array(X).flatten()
				Y=np.array(Y).flatten()
				#print(X)
				#print(Y)

				results = []

				######
				for model in models:
					result = model.predict(X)
					temp_result=0
					for i in range(len(result)):
						temp_result = temp_result+ abs(Y[i]-result[i])**2
					#print(result)
					results.append(temp_result)


				results = np.array(results)
				where_are_NaNs = np.isnan(results)
				results[where_are_NaNs] = float('inf')
				results = list(results)
				#######



				if results:
					min_index = results.index(min(results))

					new_color = correct[min_index][0].color_assigned
					cc.color_assigned=new_color

					try:
						reclassified[min_index].append(cc)
					except:
						#reclassified.insert(min_index,[cc])
						reclassified[min_index] = [cc]


		lines = []
		for key, val in reclassified.items():
			lines.append(val)
		return lines




	def get_lines_hough(self):
		'''
		theta is in degrees
		rho is in pixels

		0,0---> +y
		| x
			theta (theta is anticlockwise) xcos + ysin =rho
		|

		:return:
		'''
		img_H,img_W=self.img.shape



		rho_max = np.sqrt((img_H-1)**2 + (img_W-1)**2)
		rho_resolution = 0.2*self.AH

		#print(img_H, img_W,rho_resolution)

		max_num_rhos = np.ceil(rho_max / rho_resolution)+1


		acc_arr = np.zeros((int(max_num_rhos),10))#theta -5 - 5=10


		for cc in self.all_cc:
			if cc.area>self.cc_area_for_no_impact:
				for center in cc.centers:
					x= center.ax
					y= center.ay

					for theta in range(-5,5,1):
						rho_val = x*np.cos(theta*np.pi/180.0)+y*np.sin(theta*np.pi/180.0)

						rho_idx = int(np.ceil(rho_val / rho_resolution))

						#print(rho_idx)
						theta_idx = theta + 5

						acc_arr[rho_idx,theta_idx] = acc_arr[rho_idx,theta_idx]+1


						# try:
						# 	x_y_store[str(rho_idx)+'-'+str(theta_idx)].append((-y,x))
						# except:
						# 	x_y_store[str(rho_idx) + '-' + str(theta_idx)]=[]
						# 	x_y_store[str(rho_idx) + '-' + str(theta_idx)].append((-y,x))


		print('1:)', len(np.unique(acc_arr)))
		##############
		new_acc_arr=[]
		for i in range(0,int(max_num_rhos),10):
			temp=np.zeros(10)
			for j in range(i,i+10):
				for k in range(10):
					if j < int(max_num_rhos):
						temp[k] = temp[k] + acc_arr[j][k]


			new_acc_arr.append(temp)

		acc_arr = np.array(new_acc_arr)
		##############

		print(acc_arr)
		print('1.1:)',len(np.unique(acc_arr)))
		#print(x_y_store)
		#plot_opencv(acc_arr)

		#line in terms of ax+by+c = cos x + sin y - rho = 0
		#distance of point x,y from line
		# d = abs( (x cos + y sin -rho) )/1

		#lines from acc_array
		#rho_theta = np.nonzero(acc_arr)
		rho_theta =np.where(acc_arr > 0)

		lines = []

		print('2:)',len(rho_theta[0]))
		for i in range(len(rho_theta[0])):
			theta = rho_theta[1][i] - 5
			a= np.cos(theta*np.pi/180.0)
			b= np.sin(theta*np.pi/180.0)
			c= - rho_theta[0][i] * rho_resolution *10
			if b!=0:
				lines.append([a,b,c])

		#######################3
		#PLOTTING of lines
		image = self.img
		for line in lines:
			if line[1]!=0:
				x1= int(- line[2] / line[1])
			else:
				x1 = image.shape[1]
			y1= 0
			x2= 0
			if line[0]!=0:
				y2= int(-line[2]/line[0])
			else:
				y2= image.shape[0]

			cv.line(image, (x1,y1), (x2,y2), (100, 100), 1)

		plot_opencv(image)

		##########################


		cc_lines={}

		for cc in self.all_cc:
			ccx=  cc.gc_x
			ccy= cc.gc_y

			dists = [] #distance from lines
			for line in lines:
				dist = abs(line[0]*ccx + line[1]*ccy + line[2])
				dists.append(dist)

			#dists = np.array(dists)
			min_index = dists.index(min(dists))

			#print(dists[min_index],min_index)

			try:
				cc_lines[min_index].append(cc)
			except Exception as ex:
				cc_lines[min_index]=[cc]
				#cc_lines.append([cc])


		lines=[]
		for key,val in cc_lines.items():
			#print(key)
			lines.append(val)
			#print(len(val))

		print('3:)',len(lines))
		#print(cc_lines)
		return lines




	def get_lines_probabilistic_hough(self):
		new_img = np.zeros(self.img.shape)

		for cc in self.all_cc:
			if cc.area>self.cc_area_for_no_impact:
				for center in cc.centers:
					new_img[center.ax,center.ay] = 255


		plot_opencv(new_img)

		lines = probabilistic_hough_line(new_img, threshold=10, line_length=5,
										 line_gap=20)

		print(lines)

		for line in lines:
			p0, p1 = line
			cv.line(self.img,(p0[0], p1[0]), (p0[1], p1[1]),(100, 100), 1)
			cv.line(new_img, (p0[0], p1[0]), (p0[1], p1[1]), (100, 100), 1)


		plot_opencv(self.img)
		plot_opencv(new_img)

		return []



	def adjust_colors_regression_4(self,lines,num_components):
		reclassified = {}
		correct = []


		models = []


		for line in lines:
			components_in_line=0
			for cc in line:
				if cc.area > self.cc_area_for_no_impact:
					components_in_line=components_in_line+1

			if components_in_line > num_components:
				correct.append(line)



		for c_line in correct:
			num = len(c_line)

			CH = []
			top_x = float('inf')
			bot_x = 0
			for i in range(num):
				cc = c_line[i]
				if cc.area > self.cc_area_for_no_impact:#PARAMETER TO TUNE, same as
					if cc.topx<top_x:
						top_x=cc.topx
					if cc.topx+cc.h > bot_x:
						bot_x = cc.topx+cc.h
					for center in cc.centers:
						CH.append(center.ax)

			if CH:
				best_fit=0
				min_mse=float('inf')
				for h in range(top_x,bot_x,1):
					mse=0
					for ch in CH:
						mse= mse + abs(ch-h)**2

					if mse < min_mse:
						best_fit = h
						min_mse = mse


				if best_fit !=0:
					models.append(best_fit)


		print('No. of models',len(models))
		for trc_line in lines:
			for cc in trc_line:
				hs=[]
				for center in cc.centers:
					hs.append(center.ax)


				results = []

				######
				for model in models:
					mse = 0
					for h in hs:
						mse = mse + abs(model-h)**2

					results.append(mse)

				if results:
					min_index = results.index(min(results))

					new_color = correct[min_index][0].color_assigned
					cc.color_assigned=new_color

					try:
						reclassified[min_index].append(cc)
					except:
						#reclassified.insert(min_index,[cc])
						reclassified[min_index] = [cc]


		lines = []
		for key, val in reclassified.items():
			lines.append(val)
		return lines




	def get_lines_water_flow(self):
		method = WaterFlow(self.img)
		new_img = method.run()
		peaks, hist = projection_analysis(new_img, return_hist=True)

		self.peaks = peaks
		ccs = self.all_cc

		classified = {}

		for cc in ccs:
			hs = []
			for center in cc.centers:
				hs.append(center.ax)

			results = []

			for peak in peaks:
				mse = 0
				for h in hs:
					mse = mse + abs(peak - h) ** 2

				results.append(mse)

			if results:
				min_index = results.index(min(results))

				try:
					old_ccs=classified[min_index]

					new_color = classified[min_index][0].color_assigned
					cc.color_assigned = new_color
					classified[min_index].append(cc)
				except:
					new_color = random_color()
					cc.color_assigned = new_color
					classified[min_index] = [cc]

		lines = []
		for key, val in classified.items():
			lines.append(val)
		return lines


	def draw_peak_lines(self,img):
		for peak in self.peaks:
			cv.line(img,(0,peak),(img.shape[1],peak),(200, 200), 1)

		return img



