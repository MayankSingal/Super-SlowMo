import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model



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

	for epoch in range(5):
		for i, (imageList) in enumerate(train_loader):
			
			I0_var = torch.autograd.Variable(imageList[0]).cuda()
			I1_var = torch.autograd.Variable(imageList[-1]).cuda()

			flow_out_var = flowModel(I0_var, I1_var)

			F_0_1_x = flow_out_var[:,0,:,:]
			F_0_1_y = flow_out_var[:,1,:,:]
			F_1_0_x = flow_out_var[:,2,:,:]
			F_1_0_y = flow_out_var[:,3,:,:]
			F_0_1 = flow_out_var[:,:2,:,:]
			F_1_0 = flow_out_var[:,2:,:,:]

			loss_vector = []

			for t in range(1,8):
				It_var = torch.autograd.Variable(imageList[t]).cuda()

				F_t_0_x = -(1-t)*t*F_0_1_x + t*t*F_1_0_x
				F_t_0_y = -(1-t)*t*F_0_1_y + t*t*F_1_0_y
				F_t_1_x = (1-t)*(1-t)*F_0_1_x - t*(1-t)*(F_1_0_x)
				F_t_1_y = (1-t)*(1-t)*F_0_1_y - t*(1-t)*(F_1_0_y)

				F_t_0 = torch.cat([torch.unsqueeze(F_t_0_x,3), torch.unsqueeze(F_t_0_y,3)], 3)
				F_t_1 = torch.cat([torch.unsqueeze(F_t_1_x,3), torch.unsqueeze(F_t_1_y,3)], 3)
				
				g_I0_F_t_0 = nn.functional.grid_sample(I0_var, F_t_0, mode='bilinear')
				g_I1_F_t_1 = nn.functional.grid_sample(I1_var, F_t_1, mode='bilinear')

				# Computing again to match shape with refine network input
				F_t_0 = torch.cat([torch.unsqueeze(F_t_0_x,1), torch.unsqueeze(F_t_0_y,1)], 1)
				F_t_1 = torch.cat([torch.unsqueeze(F_t_1_x,1), torch.unsqueeze(F_t_1_y,1)], 1)

				interp_out_var = interpolationModel(I0_var, I1_var, F_0_1, F_1_0, F_t_0, F_t_1, g_I0_F_t_0, g_I1_F_t_1)
				F_t_0_final = interp_out_var[:,:2,:,:]
				F_t_1_final = interp_out_var[:,2:4,:,:]
				V_t_0 = torch.unsqueeze(interp_out_var[:,4,:,:],1)
				V_t_1 = torch.unsqueeze(interp_out_var[:,5,:,:],1)
				#print(V_t_0.size())

				g_I0_F_t_0_final = nn.functional.grid_sample(I0_var, F_t_0_final.permute(0,2,3,1), mode='bilinear')
				g_I0_F_t_1_final = nn.functional.grid_sample(I1_var, F_t_1_final.permute(0,2,3,1), mode='bilinear')

				normalization = (1-t)*V_t_0 + t*V_t_1
				interpolated_image_t_pre = (1-t)*V_t_0*g_I0_F_t_0_final + t*V_t_1*g_I0_F_t_1_final
				interpolated_image_t = interpolated_image_t_pre / normalization

				loss_t = criterion(interpolated_image_t, It_var)
				loss_vector.append(loss_t)				

			loss = sum(loss_vector)/len(loss_vector)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print(loss.data[0])
			#if (i+1 % 10) == 0:
			#	print("Loss at iteration", i, ":", loss.data[0])





						


if __name__ == '__main__':
	train_val()