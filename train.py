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

	dataFeeder = dataloader.expansionLoader('/home/user/data/nfs')
	train_loader = torch.utils.data.DataLoader(dataFeeder, batch_size=2, 
											  shuffle=True, num_workers=1,
											  pin_memory=True)
	criterion = nn.MSELoss().cuda()

	optimizer = torch.optim.Adam(list(flowModel.parameters()) + list(interpolationModel.parameters()), lr=0.0001)

	flowModel.train()
	interpolationModel.train()

	warper = FlowWarper(352,352)

	for epoch in range(5):
		for i, (imageList) in enumerate(train_loader):
			
			I0_var = torch.autograd.Variable(imageList[0]).cuda()
			I1_var = torch.autograd.Variable(imageList[-1]).cuda()
			

			flow_out_var = flowModel(I0_var, I1_var)
			#flow_out_var /= 352

			#F_0_1_x = flow_out_var[:,0,:,:]
			#F_0_1_y = flow_out_var[:,1,:,:]
			#F_1_0_x = flow_out_var[:,2,:,:]
			#F_1_0_y = flow_out_var[:,3,:,:]
			F_0_1 = flow_out_var[:,:2,:,:]
			F_1_0 = flow_out_var[:,2:,:,:]

			loss_vector = []

			for t_ in range(1,8):

				t = t_/8
				It_var = torch.autograd.Variable(imageList[t_]).cuda()

				F_t_0 = -(1-t)*t*F_0_1 + t*t*F_1_0
				#F_t_0_y = -(1-t)*t*F_0_1_y + t*t*F_1_0_y
				F_t_1 = (1-t)*(1-t)*F_0_1 - t*(1-t)*(F_1_0)
				#F_t_1_y = (1-t)*(1-t)*F_0_1_y - t*(1-t)*(F_1_0_y)

				#F_t_0 = torch.cat([torch.unsqueeze(F_t_0_x,3), torch.unsqueeze(F_t_0_y,3)], 3)
				#F_t_1 = torch.cat([torch.unsqueeze(F_t_1_x,3), torch.unsqueeze(F_t_1_y,3)], 3)
				
				g_I0_F_t_0 = warper(I0_var, F_t_0)#nn.functional.grid_sample(I0_var, F_t_0, mode='bilinear')
				g_I1_F_t_1 = warper(I1_var, F_t_1)#nn.functional.grid_sample(I1_var, F_t_1, mode='bilinear')

				# Computing again to match shape with refine network input
				#F_t_0 = torch.cat([torch.unsqueeze(F_t_0_x,1), torch.unsqueeze(F_t_0_y,1)], 1)
				#F_t_1 = torch.cat([torch.unsqueeze(F_t_1_x,1), torch.unsqueeze(F_t_1_y,1)], 1)

				interp_out_var = interpolationModel(I0_var, I1_var, F_0_1, F_1_0, F_t_0, F_t_1, g_I0_F_t_0, g_I1_F_t_1)
				F_t_0_final = interp_out_var[:,:2,:,:]
				F_t_1_final = interp_out_var[:,2:4,:,:]
				V_t_0 = torch.unsqueeze(interp_out_var[:,4,:,:],1)
				V_t_1 = 1 - V_t_0#torch.unsqueeze(interp_out_var[:,5,:,:],1)

				#print(V_t_0.size())

				g_I0_F_t_0_final = warper(I0_var, F_t_0_final)#nn.functional.grid_sample(I0_var, F_t_0_final.permute(0,2,3,1), mode='bilinear')
				g_I0_F_t_1_final = warper(I1_var, F_t_1_final)#nn.functional.grid_sample(I1_var, F_t_1_final.permute(0,2,3,1), mode='bilinear')

				normalization = (1-t)*V_t_0 + t*V_t_1
				interpolated_image_t_pre = (1-t)*V_t_0*g_I0_F_t_0_final + t*V_t_1*g_I0_F_t_1_final
				interpolated_image_t = interpolated_image_t_pre / normalization

				loss_t = criterion(interpolated_image_t, It_var)
				loss_vector.append(loss_t)				

			loss = sum(loss_vector)/len(loss_vector)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#print(loss.data[0])
			if ((i+1) % 10) == 0:
				print("Loss at iteration", i, ":", loss.data[0])
				torchvision.utils.save_image(I0_var,'1.jpg',normalize=True)
				torchvision.utils.save_image(interpolated_image_t,'2.jpg', normalize=True)
				torchvision.utils.save_image(I1_var,'3.jpg',normalize=True)





						


if __name__ == '__main__':
	train_val()