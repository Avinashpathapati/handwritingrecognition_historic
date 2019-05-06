
import cv2 as cv
import numpy as np

from recognizer.utility import plot_opencv
import math
from external_code.horizontal_profiling import get_valleys
from external_code.path_finder_a_star import Astar

from recognizer.utility import invert_image


class LineSementation():
	def __init__(self):
		pass

	def technique_hough(self,img):
		hough = HoughTransformationBased()
		hough.run(img)
		return None

	def technique_a_star(self,img):
		valleys=get_valleys(img)
		print(valleys)
		#valleys=valleys[1:3]
		img_W = img.shape[1]
		img_H = img.shape[0]
		print(img_W,img_H)

		paths=[]
		maps=[]

		for valley in valleys:
			print('working on valley--',valley)
			path_finder = Astar(grid=img)
			start_node = [int(valley),0]
			goal_node=[int(valley),img_W-1]
			start_node = (0,int(valley))
			goal_node = (img_W - 1,int(valley))

			cv.line(img,start_node,goal_node,0)
			#print(start_node,goal_node)
		# 	path_for_valley,map = path_finder.pathfind(start_node,goal_node)
		#
		# 	paths.append(path_for_valley)
		# 	maps.append(map)
		#
		# for map in maps:
		# 	self.draw_map(img,map)

		# for path in paths:
		# 	self.draw_line(img,path)


		return img

	def draw_line(self,img, path):
		for p in path:
			img[p[0], p[1]] = 0

	def draw_map(self,img, map):
		for m in map:
			img[m.row, m.col] = 0




	def test_segmentation(self,img):
		#img = self.technique_hough(img)
		img = self.technique_a_star(img)
		plot_opencv(img)
		return




class HoughTransformationBased():
	'''
	'''
	def __init__(self):
		pass


	def pre_processing(self,image):
		#Step1: Extract all the coonected componnets
		#Step2: Calculate the average character height, AW, average width is assumed same as AH.
		#Step3: Categories connected components in three sub domains.


		'''
		STEP1: CC extraction
		'''
		image=255-image #Represent white as text and background as black
		nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
		heights = stats[:, 3] #Heights of component
		widths = stats[:,2] #Widths of components

		'''
		STEP2:
		'''
		AH = sum(heights)/nb_components #nb_components is number of components
		AW = AH

		'''
		STEP3:
		'''
		#initialise image which will only contain components of subset 1 and similarly for subset 2 and three
		subset1 = np.zeros(output.shape)
		subset1_labels = np.zeros(output.shape) #array to store cc number
		subset1_all_labels = []
		subset2 = np.zeros(output.shape)
		subset2_labels = np.zeros(output.shape)
		subset2_all_labels = []
		subset3 = np.zeros(output.shape)
		subset3_labels = np.zeros(output.shape)
		subset3_all_labels = []

		for i in range(1, nb_components): #labels start from 1, i denotes label
			H = heights[i-1]
			W = widths[i-1]
			if ((0.5*AH)<=H<=(3*AH)) and ((0.5*AW)<=W): #condition defined in the paper
				subset1[output==i] = 255 #Add those components to subset1
				subset1_labels[output == i] = i
				subset1_all_labels.append(i)
			elif H>=(3*AH):
				subset2[output == i] = 255
				subset2_labels[output == i] = i
				subset2_all_labels.append(i)
			elif ((H<(3*AH)) and ((0.5*AW)>W)) or ((H<(0.5*AH)) and ((0.5*AW)<W)):
				subset3[output == i] = 255
				subset3_labels[output == i] = i
				subset3_all_labels.append(i)
			else:
				pass


		self.AH = AH #set average character height and width, which are to be used in other functions too
		self.AW = AW

		return ((subset1,subset1_labels,subset1_all_labels),(subset2,subset2_labels,subset2_all_labels),(subset3,subset3_labels,subset3_all_labels))




	def post_processing(self):
		pass


	def get_accumalator_array(self,image):
		'''
		theta is in degrees
		rho is in pixels

		0,0---> +x
		| -y
			-theta
		|

		:return:
		'''
		img_H,img_W=image.shape

		rho_max = np.sqrt((img_H-1)**2 + (img_W-1)**2)
		rho_resolution = 0.2*self.AH

		max_num_rhos = np.ceil(rho_max / rho_resolution)+1
		acc_arr = np.zeros((int(max_num_rhos),10))#theta85-94=10

		x_y_store={}

		for y in range(img_H):
			for x in range(img_W):
				if image[y][x]:
					y = -y
					for theta in range(85,95,1):
						theta = -theta
						rho_val = x*np.cos(theta*np.pi/180.0)+y*np.sin(theta*np.pi/180.0)
						rho_idx = int(np.ceil(rho_val / rho_resolution))
						theta_idx = int(-85-theta)
						acc_arr[rho_idx][theta_idx] = acc_arr[rho_idx][theta_idx]+1
						try:
							x_y_store[str(rho_idx)+'-'+str(theta_idx)].append((-y,x))
						except:
							x_y_store[str(rho_idx) + '-' + str(theta_idx)]=[]
							x_y_store[str(rho_idx) + '-' + str(theta_idx)].append((-y,x))

		#print(np.unique(acc_arr))
		#print(x_y_store)
		return acc_arr,x_y_store


	def get_boxes_within_rho_theta_area(self,rho_index,theta_index,x_y_store):
		#print(rho_index,theta_index)
		boxes=[]
		key=str(rho_index) + '-' + str(theta_index)

		all_gravity_centers =set()

		try:
			temp_gcs=x_y_store[key]
			#print(temp_gcs)
			temp_gcs = set(temp_gcs)
			all_gravity_centers=all_gravity_centers.union(temp_gcs)
			#print(all_gravity_centers)
		except Exception as ex:
			print('in exception 1')
			pass

		rho_resolution = 0.2 * self.AH
		rho_val =  rho_index*rho_resolution


		#for r in range(int(rho_val-5),int(rho_val+5),rho_resolution):
		r=rho_val-5
		while r < rho_val+5:
			r_idx = int(np.ceil(r / rho_resolution))
			key = str(r_idx) + '-' + str(theta_index)

			try:
				temp_gcs = x_y_store[key]
				temp_gcs = set(temp_gcs)
				all_gravity_centers=all_gravity_centers.union(temp_gcs)
			except Exception as ex:
				print('in exception 2')
				pass
			r=r+rho_resolution
			#print(r)

		for center in all_gravity_centers:
			#print(center)
			key = str(center[0]) + '-' + str(center[1])
			#print(key)
			try:
				box = self.gc_to_box[key]
				boxes.append(box)
			except Exception as ex:
				print('in exception 3',ex)
				pass

		#print(boxes)
		return boxes


	def is_component_part_of_area(self,component, boxes_of_rho_theta_area):
		try:
			boxes_of_component = self.box_coordinates[component]
			num_of_boxes_in_component = len(boxes_of_component)
			component_pts = []
			total_match=0
			for box in boxes_of_component:
				temp_box = box[0]#first part is tuple representing box
				gc = box[1]#second part is tuple representing gravity center coordinates of that box
				if temp_box in boxes_of_rho_theta_area:
					total_match = total_match+1
					component_pts.append(gc)

			if total_match > num_of_boxes_in_component/2:
				return True,component_pts

		except Exception as ex:
			print('in exception 4')
			pass

		return False,[]


	def get_top_rho_theta(self,acc_array):
		rho,theta = np.unravel_index(acc_array.argmax(), acc_array.shape)
		return rho,theta

	def draw_line(self,image,points):
		point1=points[0]
		for point2 in points[1:]:
			cv.line(image, point1,point2, (100, 100), 1)
			point1=point2
		#plot_opencv(image)


	def draw_lines(self,img):
		'''
		line-> top_rho_index,top_theta_index,components_in_line
		:param img:
		:return:
		'''
		#img = np.zeros(img.shape)
		rho_resolution = 0.2 * self.AH
		for line in self.lines:
			rho_index = line[0]
			theta_index = line[1]
			rho_val = rho_index * rho_resolution
			theta_val = -(85+theta_index)
			for component in line[2]:
				try:
					boxes_of_component = self.box_coordinates[component]
					for box in boxes_of_component:
						b_y_start, b_y_end, b_x_start, b_x_end = box[0]

						x1=b_x_start
						x2=b_x_end
						#y= sin_inverse( rho - xcos(theta)) * -1
						y1 = int(-(rho_val - (x1*np.cos(theta_val*np.pi/180.0)))/(np.sin(theta_val*np.pi/180.0)))
						y2 = int(-(rho_val - (x2 * np.cos(theta_val * np.pi / 180.0)))/(np.sin(theta_val*np.pi/180.0)))

						#print((x1,y1),(x2,y2))
						cv.line(img,(x1,y1),(x2,y2),100)
				except Exception as ex:
					print('in exception 5',ex)


		plot_opencv(img)




	def hough_transform_mapping(self,subset_image,label_data,all_labels):

		self.lines = [] #list of rhos and theta, representing different lines

		'''	
		Apply partitioning to each connected component in the subset. Partition into equaly sized block. width of block is AW
		'''

		'''
		STEP2 find gravity center of each partition
		'''
		gcs = self.calculate_gravity_centers_new(subset_image, label_data, all_labels)
		plot_opencv(gcs)
		flag=True
		itrans=0
		while itrans<25:
			H,x_y_store = self.get_accumalator_array(gcs)
			print('in loop', max(np.unique(H)))
			if max(np.unique(H))<5:#if number of votes is less than 5
				flag=False
				#continue

			top_rho_index,top_theta_index = self.get_top_rho_theta(H)


			#get all boxes whose gravity centers have contribution is rho-5,theta and rho+5, theta area.
			boxes_of_rho_theta_area = self.get_boxes_within_rho_theta_area(top_rho_index,top_theta_index,x_y_store)

			'''
			for each component check whether half of its point belong to this area,if yes assign it rho,theta and remove
			
			We do this by seeing if half of the boxes of the component are same as boxes_of_rho_theta_area.
			'''
			components_in_line = []
			for component in all_labels:
				#cps=np.where(label_data == component)
				flag,gcs_to_remove=self.is_component_part_of_area(component,boxes_of_rho_theta_area)
				#print(gcs_to_remove)
				if flag:
					print(component,'matched')
					all_labels.remove(component)
					#remove component from subset_image
					subset_image[label_data == component] = 0

					#remove component from label_data
					label_data[label_data==component]=0


					for temp in gcs_to_remove:
						gcs[temp[0]][temp[1]]=0

					#plot_opencv(gcs)

					components_in_line.append(component)

			# Note theta is in range 0-10 here (our specified range is 85-95 in degrees for theta),top_rho is index and actual rho we have to multiply index with resolution to get rho
			self.lines.append([top_rho_index,
							   top_theta_index,components_in_line])
			itrans=itrans+1


		return


	# def conver_nparray_to_cv_8uc1(self,arr):
	# 	h, w = arr.shape
	# 	arr = cv.CreateMat(h, w, cv.CV_8UC1)
	# 	#vis0 = cv.fromarray(vis)
	# 	return arr


	def get_gravity_center_of_block(self,block):
		# calculate moments of binary image
		M = cv.moments(block)

		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		else:
			cX, cY = 0, 0
		#print(cX,cY)
		return cX,cY

	def get_gravity_centers_for_cc(self,image,cps):
		block_width = int(self.AW)
		centers=[]
		ys = cps[0]
		xs = cps[1]
		b_y_start = min(ys)
		b_y_end = max(ys)
		block_height = b_y_end - b_y_start
		gx = min(xs)
		x_end = max(xs)
		while gx < x_end:
			b_x_start = gx
			gx = gx + block_width
			b_x_end = gx if gx < x_end else x_end
			block = image[b_y_start:b_y_end, b_x_start:b_x_end]
			# print(block)
			# print(b_x_start,b_x_end,b_y_start,b_y_end)

			cX, cY = self.get_gravity_center_of_block(block)

			centers.append((b_y_start + cY,b_x_start + cX))
		return centers



	def calculate_gravity_centers_new(self,image,label_data,all_components):
		img_H, img_W = image.shape
		img = np.zeros((img_H, img_W))  # initialise image which will contain only gravity centres
		block_width = int(self.AW)

		#also store box corordinates for each cc. and for each box store gravity center too
		self.box_coordinates = {}
		#also store gravity centers to box coordinates mapping
		self.gc_to_box={}
		for cc in all_components:
			self.box_coordinates[cc] = []
			cps = np.where(label_data == cc)
			ys = cps[0]
			xs=cps[1]
			b_y_start = min(ys)
			b_y_end = max(ys)
			block_height = b_y_end- b_y_start
			gx=min(xs)
			x_end = max(xs)
			while gx<x_end:
				b_x_start = gx
				gx=gx+block_width
				b_x_end = gx if gx < x_end else x_end
				block = image[b_y_start:b_y_end, b_x_start:b_x_end]
				# print(block)
				# print(b_x_start,b_x_end,b_y_start,b_y_end)

				cX, cY = self.get_gravity_center_of_block(block)
				if cX and cY:
					gc_y= b_y_start + cY
					gc_x = b_x_start + cX
					img[gc_y][gc_x] = 255

					self.box_coordinates[cc].append(((b_y_start,b_y_end, b_x_start,b_x_end),(gc_y,gc_x)))#

					key = str(gc_y) + '-' + str(gc_x)
					self.gc_to_box[key] = (b_y_start,b_y_end, b_x_start,b_x_end)


		return img


	def draw_blocks(self,image,label_data,all_components):
		img_H, img_W = image.shape
		block_width = int(self.AW)
		for cc in all_components:
			cps = np.where(label_data == cc)
			ys = cps[0]
			xs=cps[1]
			b_y_start = min(ys)
			b_y_end = max(ys)
			block_height = b_y_end- b_y_start
			x_start=min(xs)
			x_end = max(xs)
			gx=x_start
			cv.line(image, (x_start, b_y_start), (x_end, b_y_start), (100, 100), 1)
			while gx<x_end:

				cv.line(image,(gx,b_y_start),(gx,b_y_end),(100,100), 1)

				b_x_start = gx
				gx=gx+block_width
				b_x_end = gx if gx < x_end else x_end
				block = image[b_y_start:b_y_end, b_x_start:b_x_end]
				# print(block)
				# print(b_x_start,b_x_end,b_y_start,b_y_end)

				cX, cY = self.get_gravity_center_of_block(block)
				image[b_y_start + cY][b_x_start + cX] = 255

			cv.line(image, (x_end, b_y_start), (x_end, b_y_end), (100, 100), 1)
			cv.line(image, (x_start, b_y_end), (x_end, b_y_end), (100, 100), 1)

		return image



	def calculate_gravity_centers(self,image):
		#plot_opencv(image)
		block_size = int(self.AW)
		img_H,img_W = image.shape
		gc_H,gc_W = int(img_H/block_size),int(img_W/block_size)

		img = np.zeros((img_H,img_W)) #initialise image which will contain only gravity centres

		for gx in range(0,img_W,block_size):
			for gy in range(0,img_H,block_size):
				b_x_start = gx
				b_y_start = gy
				b_x_end = gx+block_size if gx+block_size< img_W else img_W-1
				b_y_end = gy+block_size if gy+block_size< img_H else img_H-1
				block = image[b_y_start:b_y_end,b_x_start:b_x_end]
				#print(block)
				#print(b_x_start,b_x_end,b_y_start,b_y_end)

				cX, cY = self.get_gravity_center_of_block(block)
				if cX and cY:
					img[b_y_start+cY][b_x_start+cX] = 255

		return img






	def run(self,image):
		'''

		:param image: image should be a binary image
		:return:
		'''
		subset1, subset2, subset3=self.pre_processing(image)
		#Plot all subsets to see
		plot_opencv(subset1[0])
		plot_opencv(subset2[0])
		plot_opencv(subset3[0])


		self.hough_transform_mapping(subset1[0],subset1[1],subset1[2])

		#self.draw_lines(subset1[0])

		return


