import torch
import torch.nn as nn
import math


class UNet_flow(nn.Module):

	def __init__(self,block,layers,num_classes=1000):
		super(Unet_flow, self).__init__()
		self.conv1 = nn.Conv2d(6,32,7,1,3,bias=False)
		self.conv2 = nn.Conv2d(32,32,7,1,3,bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(kernel_size=7,stride=2, padding=2)

		self.conv3 = nn.Conv2d(32,64,5,1,2,bias=False)
		self.conv4 = nn.Conv2d(64,64,5,1,2,bias=False)
		self.avgpool2 = nn.AvgPool2d(kernel_size=5,stride=2, padding=1)

		self.conv5 = nn.Conv2d(6,32,7,1,3,bias=False)
		self.conv6 = nn.Conv2d(32,32,7,1,3,bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(kernel_size=7,stride=2, padding=2)

		self.conv1 = nn.Conv2d(6,32,7,1,3,bias=False)
		self.conv2 = nn.Conv2d(32,32,7,1,3,bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(kernel_size=7,stride=2, padding=2)

		self.conv1 = nn.Conv2d(6,32,7,1,3,bias=False)
		self.conv2 = nn.Conv2d(32,32,7,1,3,bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(kernel_size=7,stride=2, padding=2)

		self.conv1 = nn.Conv2d(6,32,7,1,3,bias=False)
		self.conv2 = nn.Conv2d(32,32,7,1,3,bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(kernel_size=7,stride=2, padding=2)






