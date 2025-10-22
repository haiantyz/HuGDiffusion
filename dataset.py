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
    


