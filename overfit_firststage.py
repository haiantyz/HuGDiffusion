#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from gaussian_renderer.__init__ import render
# import open3d as o3d 
import pytorch3d 
import pytorch3d.ops
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui
import sys
from scene import Scene 
from model import PointTransformerV3
import trimesh 

from model import PV3Fit
from utils.general_utils import safe_state
import uuid
import numpy as np 
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import cv2
from PIL import Image
from utils.sh_utils import eval_sh
import torch.optim as optim
from plyfile import PlyData, PlyElement
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal
from utils.general_utils import PILtoTorch2
os.environ['CUDA_LAUNCH_BLOCKING']='1'
# from repulison import get_smallest_axis, get_normal_loss
import json
from random import randint 
from pe import get_embedder
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3+ 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    _xyz = xyz
    _features_dc = features_dc
    _features_rest = features_extra
    _opacity = opacities
    _scaling = scales
    _rotation = rots

    return _xyz, _features_dc, _features_rest, _opacity, _scaling, _rotation

def training(dataset, opt, pipe, lp, args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, person):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    print("sh degree "+ str(dataset.sh_degree))
    # scene = Scene(dataset)
    
    GSModel = PV3Fit()
    GSModel = GSModel.cuda()
    # GSModel.load_state_dict(torch.load("gsmodel0.ckpt"))
    GSModel.train()

    optimizer = optim.AdamW(GSModel.parameters(), lr=0.0003)
    scheduler = StepLR(optimizer, step_size=800, gamma=0.2)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    # first_iter += 1

    pe_layerv3, _ = get_embedder(10)
    pe_layerv1, _ = get_embedder(2)

    
    pcinput = torch.from_numpy(np.load("")).cuda().float()
    pcinputpe = pe_layerv3(pcinput)
  
    ptsnumaa = pcinputpe.shape[0]
    pccolor = torch.from_numpy(np.load("")).cuda().float()
    pccolorpe = pe_layerv3(pccolor)
  
    pc = torch.from_numpy(np.load("")).unsqueeze(0).float()
    dist, _, query_knn_pc=pytorch3d.ops.knn_points(pc,pc,K=21,return_nn=True,return_sorted=False)
    dist = torch.sqrt(dist)
    _, idxcolor, _=pytorch3d.ops.knn_points(pc,pc,K=6,return_nn=True,return_sorted=False)
    
    dgcnninput = torch.cat([query_knn_pc[:,:,1:,:]-pc.unsqueeze(2).repeat(1,1,20,1), pc.unsqueeze(2).repeat(1,1,20,1)],3).permute(0,3,1,2)
    dgcnninput = dgcnninput.cuda()
    dist = dist[...,1:]
    
    dist = dist.cuda()
    dist6nn,_,_ =pytorch3d.ops.knn_points(pc,pc,K=6,return_nn=True,return_sorted=False)
    dist6nn = torch.sqrt(dist6nn)
    dist6nn = dist6nn[...,1:].sum(-1)
    dist6nn = dist6nn/5
    dist6nn= dist6nn.cuda()
   
    distmean = dist6nn.mean()
    grid_sizee = (distmean/0.01)*0.004
   
    inputf = torch.cat([pcinputpe.unsqueeze(0), pccolorpe.unsqueeze(0)], 2).squeeze()
    
    # B, N, _ = pcinput.shape

    x = pcinput.reshape(-1, 3)  # (B * T, N, 3)
    data_dict = dict(
        feat = inputf,
        coord = x,
        grid_size = grid_sizee,
        offset = torch.arange(1, 1 + 1, device=x.device) * ptsnumaa
    )
    for epoch in range(3209):
        
        pcinput = pcinput.cuda().float()
        

        optimizer.zero_grad()
        bs = pcinput.shape[0]
        
        shs, scale, rotation, opacity = GSModel(data_dict, dist, dgcnninput)
      
        scalecolor = torch.nn.functional.normalize(scale.reshape(ptsnumaa,3))
       
        scalecolor = scalecolor
        opacitycolor = opacity.reshape(ptsnumaa,1).repeat(1,3)
        loss = 0
      
        radomangle = randint(1, 360-1)
        
        RTpath = os.path.join("the json path")
        gtimg_path = os.path.join("the gt image path")
        
        file = open(RTpath, 'r')
        js = file.read()
        dic = json.loads(js)
        RT = np.asarray(dic['RT']).reshape(3,4).transpose()*1000
        R = RT[:3]
        T = RT[-1]
        focallength = 711.111083984375
        height = 512
        width = 512
        FovY = focal2fov(focallength, height)
        FovX = FovY
        
     
        imagegt = PILtoTorch2(Image.open(gtimg_path))[:3,...].cuda()
        mask = PILtoTorch2(Image.open(gtimg_path))[3:,...].cuda()
        # print(mask.max())
        zfar = 100.0
        znear = 0.01

        trans = [0.0, 0.0, 0.0]
        scaledd = 1.0

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scaledd)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovY, fovY=FovX).transpose(0,1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).cuda()
        camera_center = world_view_transform.inverse()[3, :3].cuda()
        

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        
        render_pkg = render(pcinput[...,:3], shs[0], scale[0], rotation[0], opacity[0],  world_view_transform, FovX, full_proj_transform, camera_center, pipe, bg)
        


        image, viewspace_point_tensor, visibility_filter, radii, colors_precomp = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['colors_precomp']
        idxcolor = idxcolor.cuda()
       
        

        
        
        Ll1 = l1_loss(image*mask, imagegt*mask)
        
        if epoch %800 == 0 and epoch>0:
            
            with torch.no_grad():
                shsng, scaleng, rotationng, opacityng = GSModel(data_dict, dist, dgcnninput)

            if epoch %3200== 0:
                np.savez("the path to save the overfitted 3dgs", shs=shsng.detach().cpu().numpy(), scale = scaleng.detach().cpu().numpy(), rotation = rotationng.detach().cpu().numpy(), opacity = opacityng.detach().cpu().numpy())

            l1_test = 0.0
            psnr_test = 0.0
            for radddd in range(360):
                
                   
              
                RTpath = os.path.join("the json path")
                gtimg_path = os.path.join("the gt image path")
                file = open(RTpath, 'r')
                js = file.read()
                dic = json.loads(js)
                RT = np.asarray(dic['RT']).reshape(3,4).transpose()*1000
                R = RT[:3]
                T = RT[-1]
                focallength = 711.111083984375
                height = 512
                width = 512
                FovY = focal2fov(focallength, height)
                FovX = FovY
                
                
                imagegt = PILtoTorch2(Image.open(gtimg_path))[:3,...].cuda()
                mask = PILtoTorch2(Image.open(gtimg_path))[3:,...].cuda()
                zfar = 100.0
                znear = 0.01

                trans = [0.0, 0.0, 0.0]
                scaledd = 1.0

                world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scaledd)).transpose(0, 1).cuda()
                projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovY, fovY=FovX).transpose(0,1).cuda()
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).cuda()
                camera_center = world_view_transform.inverse()[3, :3].cuda()
                
                render_pkg = render(pcinput[...,:3], shsng[0],  scaleng[0], rotationng[0], opacityng[0], world_view_transform, FovX, full_proj_transform, camera_center, pipe, bg) #rotationng[0]  scaleng[0]
                
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(imagegt.to("cuda"), 0.0, 1.0)
                
                
                l1_test += l1_loss(image, gt_image).mean().float()
                psnr_test += psnr(image.detach()*mask.detach() , gt_image.detach()*mask.detach()).mean().float()
            print(person+": "+str(psnr_test/360))
            

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image*mask , imagegt*mask ))  + 5*torch.sqrt(torch.pow(scale-scale.mean(-2),2).mean())+((1-opacity)**2).mean() 
        print("epoch is "+str(epoch)+" is "+str(loss.item()))
        loss.backward()
        optimizer.step()
        scheduler.step()

      

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    mask = viewpoint.mask.to("cuda")
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image*mask, gt_image*mask).mean().double()
                    psnr_test += psnr(image*mask , gt_image*mask).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6030)
    parser.add_argument('--person', type=str, default='0751')
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    
    person= args.person
  
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), lp, args, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, person)

    # All done
    print("\nTraining complete.")
