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
import pytorch3d 
import pytorch3d.ops
import torch
import torch.nn as nn
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui
import sys
from scene import Scene
from model import PV3Align

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
import json
from random import randint 
from pe import get_embedder
from arguments import ModelParams, PipelineParams, OptimizationParams
from dataset import BenchmarkDatasetNewrender
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

train_dataset = BenchmarkDatasetNewrender()
    
train_data_loader = DataLoader(train_dataset, 
                    batch_size=1, shuffle=False,
                    num_workers=4, pin_memory=True)

def training(dataset, opt, pipe, lp, args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    print("sh degree "+ str(dataset.sh_degree))
    # scene = Scene(dataset)
    
    GSModel = PV3Align()
    GSModel.load_state_dict(torch.load("ckpt"))

    GSModel = GSModel.cuda()
    
    
    GSModel.eval()

    

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    

    pe_layerv3, _ = get_embedder(10)
    pe_layerv1, _ = get_embedder(2)
    
    
    for pcs, colors, shsgt, scalegt, rotationgt, opacitygt, persons in train_data_loader:
        # points, colors, shs, scale, rotation, opacity, self.persons[index]
        pcs = pcs.cuda()
        colors =colors.cuda()
        shsgt = shsgt.cuda()

        

        pcpe = pe_layerv3(pcs)
        colorpe = pe_layerv3(colors)
        bs = pcpe.shape[0]
        inputf = torch.cat([pcs, colors, pcpe, colorpe, shsgt.reshape(bs,20000,48)], 2).reshape(bs*20000, 180)
        # print(inputf.shape)
        x = pcs.reshape(-1, 3)  # (B * T, N, 3)
        data_dict = dict(
            feat = inputf,
            coord = x,
            grid_size = 0.004,
            offset = torch.arange(1, bs + 1, device=x.device) * 20000
        )
        
        
        
        with torch.no_grad():
            shs_pred = GSModel(data_dict, bs)
        print(shs_pred.shape)
        np.save("", shs = shs_pred.detach().cpu().numpy(), scale =scale.detach().cpu().numpy(),  rotation =rotation.detach().cpu().numpy(), opacity =opacity.detach().cpu().numpy(), pcs =pcs.detach().cpu().numpy())

            
        

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

    # Start GUI server, configure and run training
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), lp, args, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
