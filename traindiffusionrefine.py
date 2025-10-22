import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import MSELoss, L1Loss
from dataset import DiffREfineDataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss, SmoothL1Loss
from torch.optim.lr_scheduler import StepLR
import pytorch3d 
import pytorch3d.ops

from model.pointnet2refinement import PointNet2SemSegSSGwithPERefine

train_dataset = DiffREfineDataset()
    
train_data_loader = DataLoader(train_dataset, 
                    batch_size=6, shuffle=True,
                    num_workers=4, pin_memory=True)


mlpnet = PointNet2SemSegSSGwithPERefine()


mlpnet = mlpnet.cuda()
mlpnet.train()

optimizerG = optim.Adam(mlpnet.parameters(), lr=0.0002)
scheduler1 = StepLR(optimizerG, step_size=30, gamma=0.1)
mseloss = MSELoss()
l1loss = L1Loss()
sl1loss = SmoothL1Loss()

for epoch in range(80):
    for gspc, projectpc, projectpcback, image, backimage, imageo, backimageo, image_path, shsgt, prior_shs, _, smplidx, smpldist, smplpartcolor,scalegt,rotationgt,opacitygt in train_data_loader:
        optimizerG.zero_grad()
        bs = shsgt.shape[0]
        shsgt= shsgt.reshape(bs, 20000, 48).cuda()

        prior_shs = prior_shs.cuda()

        scalegt = scalegt.cuda()
        imageo = imageo.float().cuda()
        backimageo =backimageo.float().cuda()
      
        rotationgt = rotationgt.cuda()
        opacitygt = opacitygt.cuda()
        smpldist = smpldist.float().cuda() 
        gspc = gspc.float().cuda() 
        projectpc = projectpc.float().cuda()
        projectpcback = projectpcback.float().cuda()
        image = image.float().cuda()
        backimage = backimage.float().cuda()
        idx = smplidx.float().cuda()
        part = smplpartcolor.float().cuda()

        dist, _, query_knn_pc= pytorch3d.ops.knn_points(gspc,gspc,K=21,return_nn=True,return_sorted=False)
        dist = torch.sqrt(dist)
        # print(dist.shape)
        dgcnninput = torch.cat([query_knn_pc[:,:,1:,:]-gspc.unsqueeze(2).repeat(1,1,20,1), gspc.unsqueeze(2).repeat(1,1,20,1)],3).permute(0,3,1,2)
        
        shs_dc_off, shs_rest_pred, scale, rotation, opacity = mlpnet(gspc, projectpc, image, imageo, projectpcback, backimage, backimageo, idx, part, prior_shs, dist, dgcnninput, smpldist, True)

        predshs = torch.cat([prior_shs+shs_dc_off, shs_rest_pred], 2)
        
        
        loss = 25*l1loss(predshs, shsgt) + l1loss(scale, scalegt) + l1loss(rotation, rotationgt) + l1loss(opacity, opacitygt)

        print("epoch is + "+str(epoch) + " shs gt loss: " + str(l1loss(predshs, shsgt).item())+ " all loss: " + str(loss.item()))
        loss.backward()
        

        optimizerG.step()
        torch.cuda.empty_cache()
    scheduler1.step()
    if epoch%5==0:
        torch.save(mlpnet.state_dict(), "diff_refinenet_"+str(epoch)+".ckpt")
