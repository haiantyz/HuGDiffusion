# HuGDiffusion: Generalizable Single-Image Human Rendering via 3D Gaussian Diffusion  
## Accepted by IEEE Transactions on Visualization and Computer Graphics 2025

Please follow [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch) [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [PointTransformerV3](https://github.com/Pointcept/Pointcept) and [PVD](https://github.com/alexzhou907/PVD) to prepare the python environment.

Please follow [GuassianCube](https://gaussiancube.github.io/) or [Trellis](https://github.com/microsoft/TRELLIS) to prepare the rendered RGB images (saved as png) and corresponding camera parameters (saved as json). We provide an example folder which contains the png and json file. Note that, you might need to slightly change some codes because of the different camera parameters.

```
# first stage per-person overfitting. At this stage, you will achieve a npz file which contains 3dgs attributes.
python overfit_firststage.py

# second stage all person overfitting. At this stage, you will achieve a ckpt file first.
python unifyalign_secondstage.py

# load the ckpt file and run the inference py file you will achieve a distribution unified proxy 3dgs dataset.
python unifyalign_inference.py
```

Please follow [HaP](https://github.com/yztang4/HaP/tree/main) to generate the human point cloud. Please reffer to [instructpix2pix](https://github.com/timothybrooks/instruct-pix2pix), [controlnet](https://github.com/lllyasviel/ControlNet), [SiTH](https://github.com/SiTH-Diffusion/SiTH) and [PSHuman](https://github.com/pengHTYX/PSHuman) for generating the backside view images.
```
# train the diffusion model.
python traindiffusion.py

# train the refinement model.
python traindiffusionrefine.py

```
