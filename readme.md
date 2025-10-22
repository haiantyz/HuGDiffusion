# HuGDiffusion: Generalizable Single-Image Human Rendering via 3D Gaussian Diffusion  
## IEEE Transactions on Visualization and Computer Graphics 2025

Please follow [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch) [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [PointTransformerV3](https://github.com/Pointcept/Pointcept) and [PVD](https://github.com/alexzhou907/PVD) to prepare the python environment.

Please follow [GuassianCube](https://gaussiancube.github.io/) or [Trellis](https://github.com/microsoft/TRELLIS) to prepare the rendered RGB images (saved as png) and corresponding camera parameters (saved as json).

```
# first stage per-person overfitting. At this stage, you will achieve a npz file which contains 3dgs attributes.
python overfit_firststage.py

# second stage all person overfitting. At this stage, you will achieve a ckpt file first.
python unifyalign_secondstage.py
```

The official pytorch implement of HuGDiffusion.
