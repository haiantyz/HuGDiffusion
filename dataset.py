from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from plyfile import PlyData, PlyElement
import imageio
import torch
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import trimesh 

class BenchmarkDataset(data.Dataset):

    def __init__(self):
        self.feature_paths = []
        self.pc_paths = []
        self.color_paths = []
        self.guassianpaths = []
        self.persons = []
        
        for i in range():
            person = str(i).zfill(4)
        
            pc_path = "the file saves pcs "
            color_path = "the file saves colors"
            
            gaussian_path = "the path save the 3dgs npz file of the first stage"
            if not os.path.exists(gaussian_path):
                continue
            self.persons.append(person)
            self.pc_paths.append(pc_path)
            self.guassianpaths.append(gaussian_path)
            self.color_paths.append(color_path)
        
        
    def __getitem__(self, index):

        
        gaussian_path = self.guassianpaths[index]
        gaua = np.load(gaussian_path)
        shs = torch.from_numpy(gaua['shs']).float().squeeze()
        # print(shs.shape)
        scale = torch.from_numpy(gaua['scale']).float().squeeze()
        rotation = torch.from_numpy(gaua['rotation']).float().squeeze()
        opacity = torch.from_numpy(gaua['opacity']).float().squeeze()
        points = torch.from_numpy(np.load(self.pc_paths[index])).float()
        colors = torch.from_numpy(np.load(self.color_paths[index])).float()
        return points, colors, shs, scale, rotation, opacity, self.persons[index]

    def __len__(self):
        return len(self.persons)

class DiffREfineDataset(data.Dataset):

    def __init__(self):
        self.image_paths = []
        self.angle_paths = []
        self.pc_paths = []
        self.guassianpaths = []
        self.imageback_paths = []
        self.persons = []
        self.smplconditions = []
        self.priorshss = []
        self.scalepaths = []
        
        for i in range():
            person = str(i).zfill(4)
            for angle in range(36):
                if angle %3 !=0:
                    continue
                image_path = os.path.join("/data/thumanrender/"+person, "rendered_image_"+str(angle*10).zfill(3)+".png")

                
                target_path = "/data/backimages/"+person+"_rendered_image_"+str(angle*10).zfill(3)+".png"

                smplconditionpath = "/data/smplxregenerate/idxdistpart"+person+".npy"
                pc_path = "/data/pytorch3dthumanpoints/"+person+"_2w.ply"
                self.persons.append(person)
                self.scalepaths.append('/data/pretrained3dgs/3dgs'+person+'2w.npz')
                
                prior_path = "/data/priorshsdiff/diff_shs_"+person+"_"+str(angle)+".npy"
                gaussian_path = '/data/unify3dgsshs/'+person+".npy"
                self.image_paths.append(image_path)
                self.priorshss.append(prior_path)
                self.imageback_paths.append(target_path)
                self.pc_paths.append(pc_path)
                self.angle_paths.append(angle)
                self.guassianpaths.append(gaussian_path)
                self.smplconditions.append(smplconditionpath)

        self.rgbto_tensor = transforms.Compose([
            transforms.ToTensor(), 
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):

        pc_path = self.pc_paths[index]
        gaussian_path = self.guassianpaths[index]
        # shsgt = torch.from_numpy(np.load(gaussian_path))
        angle = self.angle_paths[index]
        gspc= torch.from_numpy(np.asarray(trimesh.load(pc_path).vertices))
        smplcondition = np.load(self.smplconditions[index])

        smplidx = torch.from_numpy(smplcondition[:,0:1])
        smpldist = torch.from_numpy(smplcondition[:,1:2])
        smplpartcolor = torch.from_numpy(smplcondition[:,2:])

        pcd = o3d.io.read_point_cloud(pc_path)
        rpo3d1 = pcd.get_rotation_matrix_from_xyz((-(np.pi/2), 0, 0))
        pcd.rotate(rpo3d1, center=(0, 0, 0))
        rpo3d = pcd.get_rotation_matrix_from_xyz((0, -int(angle)*(np.pi/18), 0))
        pcd.rotate(rpo3d, center=(0, 0, 0))
        projectpc = torch.from_numpy(np.asarray(pcd.points))
        scalegt = torch.from_numpy(np.load(self.scalepaths[index])['scale'])
        rotationgt = torch.from_numpy(np.load(self.scalepaths[index])['rotation'])
        opacitygt = torch.from_numpy(np.load(self.scalepaths[index])['opacity'])
        shsgt = torch.from_numpy(np.load(self.scalepaths[index])['shs'])
        pcdbak = o3d.io.read_point_cloud(pc_path)
        rpo3dbak = pcdbak.get_rotation_matrix_from_xyz((0, 0, -int(angle)*(np.pi/18)))
        pcdbak.rotate(rpo3dbak, center=(0, 0, 0))
        projectpcback = torch.from_numpy(np.asarray(pcdbak.points))
        # pcd = np.asarray(trimesh.load(pc_path).vertices)
        # rotatepc = VaryPoint(rotatepc, 'Z', int(angle)*10)
        # projectpcback = torch.from_numpy(rotatepc)

        image_path = self.image_paths[index]
        # print(image_path)
        backimage_path = self.imageback_paths[index]
        newsize = (1024, 1024)
        image = np.asarray(
                Image.fromarray(imageio.v2.imread(image_path))
                .convert("RGBA").resize(newsize)
            )[...,:3]/ 255.0
        # print(image_path)
        
        # rgb = rgb.resize(newsize)
        backimage = np.asarray(
                Image.fromarray(imageio.v2.imread(backimage_path))
                .convert("RGBA").resize(newsize)
            )[...,:3]/ 255.0
        imageo = np.asarray(
                Image.fromarray(imageio.v2.imread(image_path))
                .convert("RGBA")
            )[...,:3]/ 255.0
        # print(image_path)
        
        # rgb = rgb.resize(newsize)
        backimageo = np.asarray(
                Image.fromarray(imageio.v2.imread(backimage_path))
                .convert("RGBA")
            )[...,:3]/ 255.0
        # print(backimage_path)
        image = self.rgbto_tensor(image)
        backimage = self.rgbto_tensor(backimage)
        imageo = self.rgbto_tensor(imageo)
        backimageo = self.rgbto_tensor(backimageo)
        prior_shs = torch.from_numpy(np.load(self.priorshss[index]))
        return gspc, projectpc, projectpcback, image, backimage, imageo, backimageo, image_path, shsgt, prior_shs, self.persons[index], smplidx, smpldist, smplpartcolor, scalegt, rotationgt, opacitygt

    def __len__(self):
        return len(self.pc_paths)


