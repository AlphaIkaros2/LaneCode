import sys
sys.path.append("/home/pi/Brain/src/hardware/camera/threads/laneDetection.py")

import numpy as np
import cv2
import time
from sklearn.cluster import DBSCAN, k_means
from scipy import ndimage
global STEERING
STEERING = 0.0
class intersection:
	def __init__(self):
		self.au=[]
		self.bu=[]
		self.ketqua=[]
		self.b=[]
		self.dem=0
		self.w=0
		self.data=0
		self.NODE = [2,1,1,2,0,1,0,2,1,1,0]
	def kiemtra(self,line1,line2):
		if(line1[0]-line1[2]==0):
			line1[0]=line1[0]+1
		if(line2[0]-line2[2]==0):
			line2[0]=line2[0]+1
		as1=(line1[1]-line1[3])/(line1[0]-line1[2])
		bs1=line1[1]-as1*line1[0]
		as2=(line2[1]-line2[3])/(line2[0]-line2[2])
		bs2=line2[1]-as2*line2[0]
		if(((as1*line2[0]-line2[1]+bs1)*(as1*line2[2]-line2[3]+bs1)<0) and ((as2*line1[0]-line1[1]+bs2)*(as2*line1[2]-line1[3]+bs2)<0)):
			return True
		return False
	def xuly1(self,frame):
		frame_resized = cv2.resize(frame, (720, 405))
		shape = np.array([[(0, 405), (270,405), (270,250),(300,200) ,(350,190),(400,200),(450,220), (500, 250), (500, 405), (720, 405), (720,0),(0,0)]])
		cv2.fillPoly(frame_resized, shape, 0)
		# Hiển thị khung hình lên màn hình
		bgr = [200, 200, 200]
		thresh= 55
		minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
		maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
		maskBGR = cv2.inRange(frame_resized,minBGR,maxBGR)
		edges = cv2.Canny(maskBGR,2,100)
		lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)
		filtered_lines = []
		liness=[300,370,450,405]
		liness1=[300,405,450,370]
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]
				if abs(y2 - y1) < 50 and abs(x2 - x1) > 100 and (self.kiemtra(line[0],liness)==True or self.kiemtra(line[0],liness1)==True):
					filtered_lines.append(line)
		if lines is not None:
			for line in lines:
				x1,y1,x2,y2 = line[0]
				cv2.line(frame_resized,(x1,y1),(x2,y2),(0,255,0),1)
		liness=[300,370,450,405]
		liness1=[300,405,450,370]
		# vẽ đường ngang lên ảnh gốc
		cv2.line(frame_resized, (300, 370), (450, 405), (42, 55, 105), 5)          
		cv2.line(frame_resized, (300, 405), (450, 370), (42, 55, 105), 5)    
		cv2.imshow('Video', frame_resized)
		if filtered_lines is not None:
			for line in filtered_lines:
				x1, y1, x2, y2 = line[0]
				cv2.line(frame_resized, (x1, y1), (x2, y2), (42, 55, 105), 5) 
				return "True"
		else:
			return "False"
	def xuly(self,frame): 
		self.au=[0,0,0,0]
		self.bu=[0,0,0,0]
		self.b=[0]*500
		frame_resized = cv2.resize(frame, (720, 405))
		shape = np.array([[(600, 0), (600,405), (720,405), (720, 0), (600, 0)]])
		cv2.fillPoly(frame_resized, shape, 0)
		thresh=int(55)
		self.ketqua=[0,0,0]
		bgr = [200, 200, 200]
		
		minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
		maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
		maskBGR = cv2.inRange(frame_resized,minBGR,maxBGR)
		blurred = cv2.GaussianBlur(maskBGR, (3, 3), 0)
		edges = cv2.Canny(blurred,2,100)
		lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength=20,maxLineGap=10)
		if lines is not None:
			for line in lines:
				x1,y1,x2,y2 = line[0]
				cv2.line(frame_resized,(x1,y1),(x2,y2),(0,255,0),1)
		############################################################
		for i in range(170, 230, 5):
			cv2.line(frame_resized,(0,i),(250,i),(0,255,255),1)
		for i in range(200, 270, 5):
			cv2.line(frame_resized,(595,i),(370,i),(0,255,255),1)
		for i in range(270, 390, 5):
			cv2.line(frame_resized,(i,230),(i,150),(0,255,255),1)
		############################################################
		dem1=0
		if lines is not None:
			for line in lines:
				x1,y1,x2,y2 = line[0]
				self.au=[x1,y1,x2,y2]
				for i in range(170, 230, 5):
					self.bu=[0,i,250,i]
					if(self.kiemtra(self.au,self.bu)==True):
						cv2.line(frame_resized,(0,i),(250,i),(255,0,255),1)
						self.b[i]=1  
		for i in range(170, 230, 5):
			if(self.b[i]==0):
				dem1=dem1+1
		if(dem1>=4):
			self.ketqua[0]=1
		dem1=0
		self.b=[0]*500
		if lines is not None:
			for line in lines:
				x1,y1,x2,y2 = line[0]
				self.au=[x1,y1,x2,y2]
				for i in range(200, 270, 5):
					self.bu=[595,i,370,i]
					if(self.kiemtra(self.au,self.bu)==True):
						cv2.line(frame_resized,(595,i),(370,i),(255,0,255),1)
						self.b[i]=1
		for i in range(200, 270, 5):
			if(self.b[i]==0):
				dem1=dem1+1
		if(dem1>=4):
			self.ketqua[2]=1
		dem1=0
		self.b=[0]*500
		if lines is not None:
			for line in lines:
				x1,y1,x2,y2 = line[0]
				self.au=[x1,y1,x2,y2]
				for i in range(270, 390, 5):
					self.bu=[i,230,i,150]
					if(self.kiemtra(self.au,self.bu)==True):
						cv2.line(frame_resized,(i,230),(i,150),(255,0,255),1)
						self.b[i]=1
		for i in range(270, 390, 5):
			if(self.b[i]==0):
				dem1=dem1+1
		if(dem1>=4):
			self.ketqua[1]=1
		# print(self.ketqua, self.NODE[self.ketqua])
		return self.ketqua

		# Hiển thị khung hình lên màn hình
	def thucthi(self,frame):
		if self.xuly1(frame)=="True":
			x,y,z = self.xuly(frame)
			self.w=1
			if x==0 and y==0 and z==0: 
				self.w=0
		elif self.w==1:
			self.dem=self.dem+1 
			self.w=0 
		return self.dem
	def quakho(self, frame):
		data1=self.thucthi(frame)
		if(self.data==0 and self.thucthi(frame)-1>=0):
			self.data=data1 
			return self.NODE[self.thucthi(frame)-1]
		if(self.data!=data1 and self.thucthi(frame)-1>=0):
			self.data=data1
			return self.NODE[self.thucthi(frame)-1]
		return -1
def sort_by_index(list_input = [[0,1,3],[3,4,5],[2,5,6]]):
	list1 = list_input
	id = np.argsort(list1, axis=0)
	list_new = []
	for i in range(len(list1)):
		list_new.append([])
	for i,idx in enumerate(id):
		idx = idx[0]
		list_new[idx] = list1[i]
	print(list_new)
	return list_new
def decodePoint(list_left, list_middle, list_right):
	
	point_0 = [0,0]
	point_1 = [0,0]
	point_2 = [0,0]

	w_left = 0
	h_left = 0
	w_middle = 0
	h_middle = 0
	w_right = 0
	h_right = 0
	pass_accept = "left:0right:0"
	list_point = []
	const_angle = np.arctan((STEERING - 360)/105)*180/np.pi
	print("steer angle" , const_angle)
	if len(list_left) > 0:
		lenght = len(list_left)
		print("1", lenght)
		if  10 < lenght < 40:
			pass_acept = 1
		else:
			pass_acept = 0
		list_left = np.asarray(list_left)
		p = np.polyfit(list_left[:,0],list_left[:,1],1)
		angle = (np.arctan(p[0])*180)/np.pi
		point_0_min =np.min(list_left, axis=0)
		# print(point_0_min)
		point_0_max =np.max(list_left, axis=0)
		# print(point_0_max)
		point_0 = np.average(list_left, axis = 0).tolist()
		h_left = int(point_0_max[1] - point_0_min[1])
		w_left = int(point_0_max[0] - point_0_min[0])
		point_0.append(h_left)
		point_0.append(w_left)
		point_0.append(lenght)
		point_0.append(angle)
		point_0.append(pass_acept)
		list_point.append(point_0)
	
	if len(list_middle) > 0:
		lenght = len(list_middle)
		print("2", lenght)
		if  10 < lenght < 40:
			pass_acept = 1
		else:
			pass_acept = 0
		list_middle = np.asarray(list_middle)
		p = np.polyfit(list_middle[:,0],list_middle[:,1],1)
		angle = (np.arctan(p[0])*180)/np.pi

		point_1_min =np.min(list_middle, axis=0)
		# print(point_0_min)
		point_1_max =np.max(list_middle, axis=0)
		# print(point_0_max)	
		point_1 = np.average(list_middle, axis = 0).tolist()
		h_middle = int(point_1_max[1] - point_1_min[1])
		w_middle = int(point_1_max[0] - point_1_min[0])
		point_1.append(h_middle)
		point_1.append(w_middle)
		point_1.append(lenght)
		point_1.append(angle)
		point_1.append(pass_acept)
		list_point.append(point_1)
	
	if len(list_right) > 0:
		lenght = len(list_right)
		print("3", lenght)
		if  10 < lenght < 40:
			pass_acept = 1
		else:
			pass_acept = 0
		list_right = np.asarray(list_right)
		p = np.polyfit(list_right[:,0],list_right[:,1],1)
		angle = (np.arctan(p[0])*180)/np.pi
		point_2_min =np.min(list_right, axis=0)
		# print(point_0_min)
		point_2_max =np.max(list_right, axis=0)
		# print(point_0_max)
		point_2 = np.average(list_right, axis = 0).tolist()
		h_right = int(point_2_max[1] - point_2_min[1])
		w_right = int(point_2_max[0] - point_2_min[0])

		point_2.append(h_right)
		point_2.append(w_right)
		point_2.append(lenght)
		point_2.append(angle)
		point_2.append(pass_acept)
		list_point.append(point_2)
	
	angle = 0
	lenght = len(list_point)
	intersaction = "False"
	middle_x = 0
	if lenght == 0:
		return 360 , "False", pass_accept, const_angle
	elif lenght == 1:
		data = list_point[0]
		x = data[0]
		y = data[1]
		h = data[2]
		w = data[3]
		lenght = data[4]
		angle = data[5]
		if (-15 < angle < 15) and ( w > 250) and ( lenght < 400):
			intersaction = "True"
			middle_x = int(x)
		elif (-15 < angle < 15) and ( w > 250) and ( lenght > 500):
			intersaction = "False"
			middle_x = 360
		elif (x < 360) and lenght < 300 : 
			middle_x = int(x + 150)
		elif (x > 360) and lenght < 300: 
			middle_x = int(x - 150)
		else:
			middle_x = int(x)
	elif lenght == 2:
		list_point = sort_by_index(list_point)
		pass_accept = "left:" + str(list_point[0][6]) + "right:" + str(list_point[1][6]) 
		# line 1
		data = list_point[0]
		x1 = data[0]
		y1 = data[1]
		h1 = data[2]
		w1 = data[3]
		lenght1 = data[4]
		angle1 = data[5]
		
		# line 2
		data = list_point[1]
		x2 = data[0]
		y2 = data[1]
		h2 = data[2]
		w2 = data[3]
		lenght2 = data[4]
		angle2 = data[5]
		# if (-15 < angle1 < 15) and ( w1 > 250) and ( lenght1 < 400):
		# 	intersaction = "True"
		# 	middle_x = int(x1)
		# if (-15 < angle2 < 15) and ( w2 > 250) and ( lenght2 > 500):
		# 	intersaction = "True"
		# 	middle_x = int(x2)
		if  x1 < 120 and x2 <300:
			middle_x = int(x2 + 120)
		elif x1>400 and x2 > 600:
			middle_x = int(x1 - 120)
		else:
			middle_x = int((x1 + x2)/2)
	else:
		list_point = sort_by_index(list_point)
		h = list_point[1][0]
		w = list_point[1][1]
		alpha = 0
		middle_right = int((list_point[1][0] + list_point[2][0])/2)
		middle_left = int((list_point[1][0] + list_point[0][0])/2)
		if (list_point[1][0] < 360 < list_point[2][0]):
			middle_x = middle_right
			pass_accept = "left:" + str(list_point[1][6]) + "right:" + str(list_point[2][6]) 
		elif (list_point[0][0] < 360 < list_point[1][0]):
			middle_x = middle_left
			pass_accept = "left:" + str(list_point[0][6]) + "right:" + str(list_point[1][6]) 
		else:
			middle_x = 360
	print(pass_accept)
	return middle_x, intersaction, pass_accept, const_angle

def middle_point(frame, 
				bgr = [190, 190, 190], # BGR white line threahold
				thresh = 65, #Threshold
				):
	global STEERING

	THREE_LANE_TOP = int(300)
	THREE_LANE_BOT = int(405)
	HEIGHT = 405
	WIDTH = 720
	
	shape = np.array([[(30, 405), (140,250), (530,250), (630, 405),(720,405), (720, 0),(0,0),(0,405)]])
	cv2.fillPoly(frame, shape, 0)
	minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
	maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

	maskBGR = cv2.inRange(frame,minBGR,maxBGR)
	resultBGR = cv2.bitwise_or(frame, frame, mask = maskBGR)

	resultBGR = cv2.resize(resultBGR, (WIDTH, HEIGHT))

	grayImage = cv2.cvtColor(resultBGR, cv2.COLOR_BGR2GRAY)

	(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 140, 255, cv2.THRESH_BINARY)
	three_lane_image = blackAndWhiteImage[THREE_LANE_TOP :THREE_LANE_BOT, 0:WIDTH]
	
	indices = np.where(three_lane_image!= [0])
	coordinates = zip(indices[1], indices[0])
	list_three_lane_point = list(coordinates)[::30]
	list_left = []
	list_middle = []
	list_right = []

	data = np.asarray(list_three_lane_point)
	if len(data) > 0:
		clustering = DBSCAN(eps=70, min_samples=3).fit(data)
		labels = clustering.labels_
		# print(clustering.labels_)
		for id,point in enumerate(list_three_lane_point):
			if labels[id] == 0:
				list_left.append(point)
				cv2.circle(resultBGR,(point[0], point[1] + 	THREE_LANE_TOP),1,(15,244,30), thickness=3, lineType=cv2.LINE_AA)
			elif labels[id] == 1:
				list_middle.append(point)
				cv2.circle(resultBGR,(point[0], point[1] + THREE_LANE_TOP),1,(255,0,0), thickness=3, lineType=cv2.LINE_AA)
			elif labels[id] == 2:
				list_right.append(point)
				cv2.circle(resultBGR,(point[0], point[1] + THREE_LANE_TOP),1,(10,41,220), thickness=3, lineType=cv2.LINE_AA)
	
	middle_x, intersaction, pass_accept, angle = decodePoint(list_left, list_middle, list_right)
	# print("middle:",middle_x)
	# print("intersaction:",intersaction)
	# print("pass:", pass_accept)
	# print("////////////////////////////////")
	cv2.circle(resultBGR,(middle_x, 300),10,(123,155,200), thickness=3, lineType = cv2.LINE_AA)
	cv2.line(resultBGR,(0,THREE_LANE_TOP), (WIDTH,THREE_LANE_TOP),(113,222,255),thickness=1,lineType=cv2.LINE_AA)
	cv2.line(resultBGR,(0,THREE_LANE_BOT), (WIDTH,THREE_LANE_BOT),(113,223,255),thickness=1,lineType=cv2.LINE_AA)
	STEERING = middle_x
	# cv2.imshow('Black white image', blackAndWhiteImage)
	# cv2.imshow('detect',resultBGR)
	# frame = cv2.resize(frame, (WIDTH,HEIGHT))
	# cv2.imshow("frame", frame)	
	# cv2.waitKey(0)

	return angle
	# return middle_point, intersection, pass_accept

if __name__ == "__main__":
	cap = cv2.VideoCapture("/home/pi/Downloads/lane.mp4")
	STEERING = 0
	intersection_check = intersection() 
	NODE = [2,1,1,2,0,1,0,2,1,1,0]
	# while True:
	# 	t1 = time.time()
	# 	_, frame = cap.read()
	# 	frame = cv2.resize(frame, (720,405))
	# 	# middle_point(frame)
	# 	cv2.imshow("frames", frame)
	# 	cv2.waitKey(0)
	# 	print(1/(time.time() - t1))
	while cap.isOpened():
		ret, frame = cap.read()
		frame = cv2.resize(frame, (720,405))
		if ret == True:
			middle_point(frame)
			# cv2.imshow("Frame", frame)
			if cv2.waitKey(1) == ord('q'):
				break
		else:
			break
	cap.release()
	cv2.destroyAllWindows() 


