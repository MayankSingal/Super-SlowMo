import torch
import torch.nn as nn
import math


class UNet_flow(nn.Module):

	def __init__(self,block,layers,num_classes=1000):
		super(Unet_flow, self).__init__()

		## Encoder 
		self.conv1 = nn.Conv2d(6,32,7,1,3,bias=False)
		self.conv2 = nn.Conv2d(32,32,7,1,3,bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool1 = nn.AvgPool2d(kernel_size=7,stride=2, padding=2)

		self.conv3 = nn.Conv2d(32,64,5,1,2,bias=False)
		self.conv4 = nn.Conv2d(64,64,5,1,2,bias=False)
		self.avgpool2 = nn.AvgPool2d(kernel_size=5,stride=2, padding=1)

		self.conv5 = nn.Conv2d(64,128,3,1,1,bias=False)
		self.conv6 = nn.Conv2d(128,128,3,1,1,bias=False)
		#self.relu = nn.ReLU(inplace=True)
		self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=2, padding=1)

		self.conv7 = nn.Conv2d(128,256,3,1,1,bias=False)
		self.conv8 = nn.Conv2d(256,256,3,1,1,bias=False)
		#self.relu = nn.ReLU(inplace=True)
		self.avgpool4 = nn.AvgPool2d(kernel_size=3,stride=2, padding=1)

		self.conv9 = nn.Conv2d(256,512,3,1,1,bias=False)
		self.conv10 = nn.Conv2d(512,512,3,1,1,bias=False)
		#self.relu = nn.ReLU(inplace=True)
		self.avgpool5 = nn.AvgPool2d(kernel_size=3,stride=2, padding=2)

		self.conv11 = nn.Conv2d(512,512,3,1,1,bias=False)
		self.conv12 = nn.Conv2d(512,512,3,1,1,bias=False)
		#self.relu = nn.ReLU(inplace=True)
		

		## Decoder
		self.upsample2D = nn.Upsample(scale_factor=2, mode='bilinear')

		self.conv13 = nn.Conv2d(512,512,3,1,1,bias=False)
		self.conv14 = nn.Conv2d(512,512,3,1,1,bias=False)

		self.conv15 = nn.Conv2d(512,256,3,1,1,bias=False)
		self.conv16 = nn.Conv2d(256,256,3,1,1,bias=False)

		self.conv17 = nn.Conv2d(256,128,3,1,1,bias=False)
		self.conv18 = nn.Conv2d(128,128,3,1,1,bias=False)

		self.conv19 = nn.Conv2d(128,64,3,1,1,bias=False)
		self.conv20 = nn.Conv2d(128,64,3,1,1,bias=False)

		self.conv21 = nn.Conv2d(64,32,3,1,1,bias=False)
		self.conv22 = nn.Conv2d(32,32,3,1,1,bias=False)

		self.conv23 = nn.Conv2d(32,6,3,1,1,bias=False)
		
		self.tanh = nn.Tanh()




	def forward(self, I0, I1):
		sources = []


		X = torch.cat([I0, I1], axis=1)

		## Encoder
		X = self.conv1(X)
		X = self.conv2(X)
		X = self.relu(X)
		sources.append(X)

		X = self.avgpool1(X)
		X = self.conv3(X)
		X = self.conv4(X)
		X = self.relu(X)
		sources.append(X)

		X = self.avgpool2(X)
		X = self.conv5(X)
		X = self.conv6(X)
		X = self.relu(X)
		sources.append(X)

		X = self.avgpool3(X)
		X = self.conv7(X)
		X = self.conv8(X)
		X = self.relu(X)
		sources.append(X)

		X = self.avgpool4(X)
		X = self.conv9(X)
		X = self.conv10(X)
		X = self.relu(X)
		sources.append(X)

		X = self.avgpool5(X)
		X = self.conv11(X)
		X = self.conv12(X)
		X = self.relu(X)
		

		## Decoder
		X = self.upsample2D(X)
		X = self.conv13(X)
		X = X + sources[-1]
		X = self.conv14(X)
		X = self.relu(X)

		X = self.upsample2D(X)
		X = self.conv15(X)
		X = X + sources[-2]
		X = self.conv16(X)
		X = self.relu(X)

		X = self.upsample2D(X)
		X = self.conv17(X)
		X = X + sources[-3]
		X = self.conv18(X)
		X = self.relu(X)

		X = self.upsample2D(X)
		X = self.conv19(X)
		X = X + sources[-4]
		X = self.conv20(X)
		X = self.relu(X)

		X = self.upsample2D(X)
		X = self.conv21(X)
		X = X + sources[-5]
		X = self.conv22(X)
		X = self.relu(X)

		X = self.conv23(X)
		out = self.tanh(X)

		return out




