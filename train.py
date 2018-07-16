import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np

class FlowWarper(nn.Module):
    def __init__(self, w, h):
        super(FlowWarper, self).__init__()
        x = np.arange(0,w)
        y = np.arange(0,h)
        gx, gy = np.meshgrid(x,y)
        self.w = w
        self.h = h
        self.grid_x = torch.autograd.Variable(torch.Tensor(gx), requires_grad=False).cuda()
        self.grid_y = torch.autograd.Variable(torch.Tensor(gy), requires_grad=False).cuda()

    def forward(self, img, uv):
        u = uv[:,0,:,:]
        v = uv[:,1,:,:]
        X = self.grid_x.unsqueeze(0).expand_as(u) + u
        Y = self.grid_y.unsqueeze(0).expand_as(v) + v
        X = 2*(X/self.w - 0.5)
        Y = 2*(Y/self.h - 0.5)
        grid_tf = torch.stack((X,Y), dim=3)
        img_tf = torch.nn.functional.grid_sample(img, grid_tf)
        return img_tf


def train_val():

	#cudnn.benchmark = True
	flowModel = model.UNet_flow().cuda()
	interpolationModel = model.UNet_refine().cuda()

	### ResNet for Perceptual Loss
	res50_model = torchvision.models.resnet18(pretrained=True)
	res50_conv = nn.Sequential(*list(res50_model.children())[:-2])
	res50_conv.cuda()

	for param in res50_conv.parameters():
		param.requires_grad = False


	#dataFeeder = dataloader.expansionLoader('/home/user/data/nfs')
	dataFeeder = dataloader.expansionLoader('/home/user/data/original_high_fps_videos')
	train_loader = torch.utils.data.DataLoader(dataFeeder, batch_size=2, 
											  shuffle=True, num_workers=1,
											  pin_memory=True)
	criterion = nn.L1Loss().cuda()
	criterionMSE = nn.MSELoss().cuda()

	optimizer = torch.optim.Adam(list(flowModel.parameters()) + list(interpolationModel.parameters()), lr=0.0001)

	flowModel.train()
	interpolationModel.train()

	warper = FlowWarper(352,352)

	for epoch in range(5):
		for i, (imageList) in enumerate(train_loader):
			
			I0_var = torch.autograd.Variable(imageList[0]).cuda()
			I1_var = torch.autograd.Variable(imageList[-1]).cuda()
			#torchvision.utils.save_image((I0_var),'samples/'+ str(i+1) +'1.jpg',normalize=True)
			#brak	


			flow_out_var = flowModel(I0_var, I1_var)
			
			F_0_1 = flow_out_var[:,:2,:,:]
			F_1_0 = flow_out_var[:,2:,:,:]

			loss_vector = []
			perceptual_loss_collector = []
			warping_loss_collector = []

			image_collector = []
			for t_ in range(1,8):

				t = t_/8
				It_var = torch.autograd.Variable(imageList[t_]).cuda()

				F_t_0 = -(1-t)*t*F_0_1 + t*t*F_1_0
				
				F_t_1 = (1-t)*(1-t)*F_0_1 - t*(1-t)*(F_1_0)
				
				
				g_I0_F_t_0 = warper(I0_var, F_t_0)
				g_I1_F_t_1 = warper(I1_var, F_t_1)

				interp_out_var = interpolationModel(I0_var, I1_var, F_0_1, F_1_0, F_t_0, F_t_1, g_I0_F_t_0, g_I1_F_t_1)
				F_t_0_final = interp_out_var[:,:2,:,:] + F_t_0
				F_t_1_final = interp_out_var[:,2:4,:,:] + F_t_1
				V_t_0 = torch.unsqueeze(interp_out_var[:,4,:,:],1)
				V_t_1 = 1 - V_t_0

				g_I0_F_t_0_final = warper(I0_var, F_t_0_final)
				g_I0_F_t_1_final = warper(I1_var, F_t_1_final)

				normalization = (1-t)*V_t_0 + t*V_t_1
				interpolated_image_t_pre = (1-t)*V_t_0*g_I0_F_t_0_final + t*V_t_1*g_I0_F_t_1_final
				interpolated_image_t = interpolated_image_t_pre / normalization
				image_collector.append(interpolated_image_t)

				### Reconstruction Loss Collector ###
				loss_reconstruction_t = criterion(interpolated_image_t, It_var)
				loss_vector.append(loss_reconstruction_t)		

				### Perceptual Loss Collector ###
				feat_pred = res50_conv(interpolated_image_t)
				feat_gt = res50_conv(It_var)
				loss_perceptual_t = criterionMSE(feat_pred, feat_gt)
				perceptual_loss_collector.append(loss_perceptual_t)

				### Warping Loss Collector ###
				g_I0_F_t_0_i = warper(I0_var, F_t_0)
				g_I1_F_t_1_i = warper(I1_var, F_t_1)
				loss_warping_t = criterion(g_I0_F_t_0_i, It_var) + criterion(g_I1_F_t_1_i, It_var)
				warping_loss_collector.append(loss_warping_t)

			### Reconstruction Loss Computation ###	
			loss_reconstruction = sum(loss_vector)/len(loss_vector)

			### Perceptual Loss Computation ###
			loss_perceptual = sum(perceptual_loss_collector)/len(perceptual_loss_collector)

			### Warping Loss Computation ###
			g_I0_F_1_0 = warper(I0_var, F_1_0)
			g_I1_F_0_1 = warper(I1_var, F_0_1)
			loss_warping = (criterion(g_I0_F_1_0, I1_var) + criterion(g_I1_F_0_1, I0_var)) + sum(warping_loss_collector)/len(warping_loss_collector) 

			### Smoothness Loss Computation ###
			loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
			loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
			loss_smooth = loss_smooth_1_0 + loss_smooth_0_1


			### Overall Loss
			loss = 0.8*loss_reconstruction + 0.005*loss_perceptual + 0.4*loss_warping + loss_smooth

			### Optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if ((i+1) % 10) == 0:
				print("Loss at iteration", i+1, "/", len(train_loader), ":", loss.item())
			
			if ((i+1) % 100) == 0:
				torchvision.utils.save_image((I0_var),'samples/'+ str(i+1) +'1.jpg',normalize=True)
				for jj,image in enumerate(image_collector):
					torchvision.utils.save_image((image),'samples/'+ str(i+1) + str(jj+1)+'.jpg',normalize=True)
				torchvision.utils.save_image((I1_var),'samples/'+str(i+1)+'9.jpg',normalize=True)





						


if __name__ == '__main__':
	train_val()
