import cv2 as cv
import numpy as np



class BoundingBox(object):
	def __init__(self,topx,topy,h,w,image_portion,label_portion):
		self.topx=topx
		self.topy = topy

		self.h = h
		self.w = w

		self.image_portion=image_portion
		self.label_portion = label_portion


		self.filter_only_one_component()

		self.upper_boundary_pixels = []
		self.lower_boundary_pixels = []

		self.middle_boundary_pixels = []

		self.water_flow_image = np.zeros(shape=self.image_portion.shape)

		self.angle = 15


	def filter_only_one_component(self):
		unique_labels = np.unique(self.label_portion) # all different type of labels in image portion

		unique_labels = list(unique_labels)
		if 0 in unique_labels:
			unique_labels.remove(0)

		if len(unique_labels)<2:
			#print('Only one label')
			self.label = unique_labels[0]
			return

		flattened = self.label_portion
		flattened = flattened.flatten()
		if 0 in flattened:
			flattened=list(flattened)
			flattened = [x for x in flattened if x != 0]
			flattened=np.array(flattened)

		y = np.bincount(flattened)
		ii = np.nonzero(y)[0]
		sorted_vals = sorted(zip(ii, y[ii]),key=lambda x:x[1])
		max= sorted_vals[-1][0]

		self.image_portion[self.label_portion != max ] = 255
		self.label = max # this will be the label of major connected component in the box


	def set_pixel_types(self):
		for row in range(1,self.image_portion.shape[0]-1):
			for col in range(1,self.image_portion.shape[1]-1):
				pixel =  self.image_portion[row,col]
				upper = self.image_portion[row-1,col]
				lower = self.image_portion[row+1,col]
				side0= self.image_portion[row-1,col+1]
				side1=self.image_portion[row,col+1]
				side2=self.image_portion[row+1,col+1]

				side10 = self.image_portion[row - 1, col - 1]
				side11 = self.image_portion[row, col - 1]
				side12 = self.image_portion[row + 1, col - 1]
				#side2=255

				#Upper boundary
				if pixel==0 and upper == 255 and lower == 0 and side0 == 255 and side1 == 255:# and side2 == 255:
					self.upper_boundary_pixels.append((row,col))

				#Lower boundary
				elif pixel==0 and upper == 0 and lower == 255 and side2 == 255 and side1 == 255:# and side0 == 255:
					self.lower_boundary_pixels.append((row,col))

				# #middle boundary
				#
				# elif pixel==0 and upper == 255 and lower == 255 and side0 == 255 and side1 == 255 and side2 == 255:
				# 	print('Got middle boundary')
				# 	self.middle_boundary_pixels.append((row,col))

				elif pixel==0 and upper == 255 and lower == 0 and side10 == 255 and side11 == 255:# and side12 == 255:
					self.upper_boundary_pixels.append((row,col))

				elif pixel==0 and upper == 0 and lower == 255 and side12 == 255 and side11 == 255:# and side10 == 255:
					self.lower_boundary_pixels.append((row,col))



	def start_water_flow(self):
		mask1=self.water_flow_left_to_right_2()
		mask2=self.water_flow_right_to_left_2()

		overall = np.zeros(shape=self.image_portion.shape)
		#overall = cv.bitwise_and(mask1,mask2)

		overall[mask1 == 255] = 255
		overall[mask2 == 255] = 255

		self.water_flow_image=overall


	def water_flow_left_to_right_2(self):
		temp_masks_upper = []
		for i in range(len(self.upper_boundary_pixels)):
			temp_mask_upper = np.zeros(self.image_portion.shape)
			pixel = self.upper_boundary_pixels[i]
			r = pixel[0]
			c = pixel[1]

			for j in range(r,self.image_portion.shape[0],1):
				for k in range(c,self.image_portion.shape[1],1):
					if ((j-r) - (np.tan(self.angle* (np.pi/180)))*(k-c)) > 0:
						temp_mask_upper[j,k]=255

			temp_masks_upper.append(temp_mask_upper)

		#Union
		mask_upper = np.zeros(shape=self.image_portion.shape)
		for i in range(len(self.upper_boundary_pixels)):
			mask_upper[temp_masks_upper[i] == 255] = 255

		temp_masks_lower = []
		for i in range(len(self.lower_boundary_pixels)):
			temp_mask_lower = np.zeros(self.image_portion.shape)
			pixel = self.lower_boundary_pixels[i]
			r = pixel[0]
			c = pixel[1]

			for j in range(r,0,-1):
				for k in range(c,self.image_portion.shape[1],1):
					if ((r-j) - (np.tan(self.angle* (np.pi/180)))*(k-c)) > 0:
						temp_mask_lower[j,k]=255

			temp_masks_lower.append(temp_mask_lower)

		# Union
		mask_lower = np.zeros(shape=self.image_portion.shape)
		for i in range(len(self.lower_boundary_pixels)):
			mask_lower[temp_masks_lower[i] == 255] = 255


		# overall = np.zeros(shape=self.image_portion.shape)
		#
		# overall[mask_upper == 255 and mask_lower == 255] = 255

		overall= cv.bitwise_and(mask_upper, mask_lower)

		return overall

	def water_flow_right_to_left_2(self):
		temp_masks_upper = []
		for i in range(len(self.upper_boundary_pixels)):
			temp_mask_upper = np.zeros(self.image_portion.shape)
			pixel = self.upper_boundary_pixels[i]
			r = pixel[0]
			c = pixel[1]

			for j in range(r, self.image_portion.shape[0], 1):
				for k in range(c,0,-1):
					if ((j-r) - (np.tan(self.angle * (np.pi / 180))) * (c-k)) > 0:
						temp_mask_upper[j,k] = 255

			temp_masks_upper.append(temp_mask_upper)

		# Union
		mask_upper = np.zeros(shape=self.image_portion.shape)
		for i in range(len(self.upper_boundary_pixels)):
			mask_upper[temp_masks_upper[i] == 255] = 255

		temp_masks_lower = []
		for i in range(len(self.lower_boundary_pixels)):
			temp_mask_lower = np.zeros(self.image_portion.shape)
			pixel = self.lower_boundary_pixels[i]
			r = pixel[0]
			c = pixel[1]

			for j in range(r, 0, -1):
				for k in range(c,0,-1):
					if ((r-j) - (np.tan(self.angle * (np.pi / 180))) * (c-k)) > 0:
						temp_mask_lower[j,k] = 255

			temp_masks_lower.append(temp_mask_lower)

		# Union
		mask_lower = np.zeros(shape=self.image_portion.shape)
		for i in range(len(self.lower_boundary_pixels)):
			mask_lower[temp_masks_lower[i] == 255] = 255

		# overall = np.zeros(shape=self.image_portion.shape)
		#
		# overall[mask_upper == 255 and mask_lower == 255] = 255

		overall = cv.bitwise_and(mask_upper, mask_lower)

		return overall




	def water_flow_left_to_right(self):
		for pixel in self.upper_boundary_pixels:
			r=pixel[0]
			c=pixel[1]
			try:
				self.water_flow_image[r,c]=255
			except:
				pass
			try:
				self.water_flow_image[r+1,c]=255
			except:
				pass
			try:
				self.water_flow_image[r+1, c+1] = 255
			except:
				pass
			try:
				self.water_flow_image[r+1, c+2] = 255
			except:
				pass
			try:
				self.water_flow_image[r+1, c+3] = 255
			except:
				pass
			try:
				self.water_flow_image[r+1, c+4] = 255
			except:
				pass

		for pixel in self.lower_boundary_pixels:
			r = pixel[0]
			c = pixel[1]
			try:
				self.water_flow_image[r, c] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c + 1] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c + 2] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c + 3] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c + 4] = 255
			except:
				pass

	def water_flow_right_to_left(self):
		for pixel in self.upper_boundary_pixels:
			r = pixel[0]
			c = pixel[1]
			try:
				self.water_flow_image[r, c] = 255
			except:
				pass
			try:
				self.water_flow_image[r + 1, c] = 255
			except:
				pass
			try:
				self.water_flow_image[r + 1, c - 1] = 255
			except:
				pass
			try:
				self.water_flow_image[r + 1, c - 2] = 255
			except:
				pass
			try:
				self.water_flow_image[r + 1, c - 3] = 255
			except:
				pass
			try:
				self.water_flow_image[r + 1, c - 4] = 255
			except:
				pass

		for pixel in self.lower_boundary_pixels:
			r = pixel[0]
			c = pixel[1]

			try:
				self.water_flow_image[r, c] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c - 1] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c - 2] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c - 3] = 255
			except:
				pass
			try:
				self.water_flow_image[r - 1, c - 4] = 255
			except:
				pass




class WaterFlow(object):
	def __init__(self,img,angle=5):
		self.img = img
		self.angle = angle  # water flow angle in degrees

		self.all_boxes=[]



	def start_water_flow_algorithm(self,angle):
		pass


	def set_bounding_boxes(self):
		#######
		# Bounding box detection
		#######
		image = 255 - self.img
		nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
		left_cols = stats[:, 0]
		top_rows = stats[:, 1]
		widths = stats[:, 2]
		heights = stats[:, 3]

		for i in range(1, nb_components):
			topx = top_rows[i]
			topy = left_cols[i]
			h = heights[i]
			w = widths[i]
			# print(topx,topy,h,w)
			image_portion = self.img[topx:topx + h, topy:topy + w]
			image_portion = np.pad(image_portion, ((2,2),(2,2)), 'constant')
			label_portion = output[topx:topx + h, topy:topy + w]
			label_portion = np.pad(label_portion, ((2, 2), (2, 2)), 'constant')
			bb = BoundingBox(topx-2, topy-2, h+4, w+4, image_portion, label_portion)

			self.all_boxes.append(bb)


	def print_labels_of_bounding_boxes(self):
		for bb in self.all_boxes:
			print(bb.label)
			print(bb.upper_boundary_pixels,bb.lower_boundary_pixels,bb.middle_boundary_pixels)


	def set_pixel_types_in_bb(self):
		for bb in self.all_boxes:
			bb.set_pixel_types()



	def start_water_flow(self):
		for bb in self.all_boxes:
			bb.start_water_flow()


	def get_final_mask(self):
		#img = np.zeros((self.img.shape[0]+4,self.img.shape[1]+4))
		img = np.zeros(self.img.shape)
		for bb in self.all_boxes:
			pixels_of_white = np.where(bb.water_flow_image == 255)
			row_vals = pixels_of_white[0] + bb.topx
			col_vals = pixels_of_white[1] + bb.topy

			for r,c in zip(row_vals,col_vals):
				try:
					img[r,c]=255
				except:
					pass

		return img

	def draw_blocks(self,img):
		image=img
		for bb in self.all_boxes:
			x_start = bb.topy
			x_end = bb.topy+bb.w
			y_start=bb.topx
			y_end = bb.topx + bb.h

			#print(x_start,y_start,x_end,y_end)
			cv.line(image, (x_start, y_start), (x_end, y_start), (100, 100), 1)
			cv.line(image, (x_end, y_start), (x_end, y_end), (100, 100), 1)
			cv.line(image, (x_end, y_end), (x_start, y_end), (100, 100), 1)
			cv.line(image, (x_start, y_end), (x_start, y_start), (100, 100), 1)

		return image


	def run(self):
		self.set_bounding_boxes()
		###########

		#return self.draw_blocks(self.img)
		############
		#Step2:
		self.set_pixel_types_in_bb()
		##############

		self.start_water_flow()

		img = self.get_final_mask()

		return img

		#self.print_labels_of_bounding_boxes()



