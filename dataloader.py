import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import glob
import os
import random


def populateTrainList(folderPath):
	folderList_pre = [x[0] for x in os.walk(folderPath)]
	folderList = []
	trainList = []

	for folder in folderList_pre:
		if folder[-3:] == '240':
			folderList.append(folder + "/" + folder.split("/")[-2])


	for folder in folderList:
		imageList = sorted(glob.glob(folder + '/' + '*.jpg'))
		for i in range(0, len(imageList), 12):
			tmp = imageList[i:i+12]
			if len(tmp) == 12:
			    trainList.append(imageList[i:i+12])

	
	return trainList

def populateTrainList2(folderPath):
	folderList = [x[0] for x in os.walk(folderPath)]
	trainList = []

	for folder in folderList:
		imageList = sorted(glob.glob(folder + '/' + '*.jpg'))
		for i in range(0, len(imageList), 12):
			tmp = imageList[i:i+12]
			if len(tmp) == 12:
			    trainList.append(imageList[i:i+12])
	return trainList





def randomCropOnList(image_list, output_size):
	
	cropped_img_list = []

	h,w = output_size
	height, width, _ = image_list[0].shape

	#print(h,w,height,width)

	i = random.randint(0, height - h)
	j = random.randint(0, width - w)

	st_y = 0
	ed_y = w
	st_x = 0
	ed_x = h

	or_st_y = i 
	or_ed_y = i + w
	or_st_x = j
	or_ed_x = j + h    

	#print(st_x, ed_x, st_y, ed_y)
	#print(or_st_x, or_ed_x, or_st_y, or_ed_y)


	for img in image_list:
		new_img = np.empty((h,w,3), dtype=np.float32)
		new_img.fill(128)
		new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()
		cropped_img_list.append(np.ascontiguousarray(new_img))


	return cropped_img_list



#print(len(populateTrainList('/home/user/data/nfs/')))

class expansionLoader(data.Dataset):

	def __init__(self, folderPath):

		self.trainList = populateTrainList2(folderPath)
		print("# of training samples:", len(self.trainList))


	def __getitem__(self, index):

		img_path_list = self.trainList[index]
		start = random.randint(0,3)
		h,w,c = cv2.imread(img_path_list[0]).shape

		image = cv2.cv2.imread(img_path_list[0])
		
		#print(h,w,c)

		if h > w:
			scaleX = int(360*(h/w))
			scaleY = 360
		elif h <= w:
			scaleX = 360
			scaleY = int(360*(w/h))



		img_list = []

		flip = random.randint(0,1)
		if flip:
			for img_path in img_path_list[start:start+9]:
				tmp = cv2.resize(cv2.imread(img_path), (scaleX,scaleY))[:,:,(2,1,0)]
				img_list.append(np.array(cv2.flip(tmp,1), dtype=np.float32))
		else:
			for img_path in img_path_list[start:start+9]:
				tmp = cv2.resize(cv2.imread(img_path), (scaleX, scaleY))[:,:,(2,1,0)]
				img_list.append(np.array(tmp,dtype=np.float32))
		#cv2.imshow("j",tmp)
		#cv2.waitKey(0) & 0xff
		#brak
		for i in range(len(img_list)):
			#print(img_list[i].shape)
			#brak
			img_list[i] /= 255
			img_list[i][:,:,0] -= 0.485#(img_list[i]/127.5) - 1
			img_list[i][:,:,1] -= 0.456
			img_list[i][:,:,2] -= 0.406

			img_list[i][:,:,0] /= 0.229
			img_list[i][:,:,1] /= 0.224
			img_list[i][:,:,2] /= 0.225

		cropped_img_list = randomCropOnList(img_list,(352,352))
		for i in range(len(cropped_img_list)):
			cropped_img_list[i] = torch.from_numpy(cropped_img_list[i].transpose((2, 0, 1)))

		
		return cropped_img_list  


	def __len__(self):
		return len(self.trainList)

