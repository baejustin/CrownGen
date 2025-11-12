# CrownGen: Patient-customized Crown Generation via Point Diffusion Model

## Requirements

This repository was tested using the following environment.

```
python==3.11.0
torch==2.5.1
pytorch-lightning==2.5.6
pytorch3d==0.7.8     
torchvision==0.20.1
matplotlib==3.8.3
open3d==0.19.0
trimesh=4.6.2
ipdb==0.13.13
ninja==1.11.1.3
numpy==1.26.4
torch-scatter==2.1.2
opencv-python==4.11.0.86
libigl==2.6.1
omegaconf==2.3.0
scipy==1.12.0
```

## Data

Both the boundary prediction module and the crown generation module fetch dentition data from the `data/dentition` directory. The following is the data structure for each dentition. Sample dentition data is included for reference.
```
DATA0001  # patient_id
│  
├─ verts  # point clouds of individual teeth, labeled by their FDI numbers  
│  │  
│  ├─ upper_FDI011.npy
│  ├─ upper_FDI012.npy
│  ├─ ...
│  ├─ lower_FDI031.npy
│  ├─ lower_FDI032.npy
│  ├─ ...
│  
├─ boundary  # cylindrical boundaries of each tooth
│  │  
│  ├─ upper_FDI011_boundary.json
│  ├─ upper_FDI012_boundary.json
│  ├─ ...
│  ├─ lower_FDI031_boundary.json
│  ├─ lower_FDI032_boundary.json
│  ├─ ...
│  
```
With your own data, specify the dentition splits in `data/splits_boundpred.json` and `data/splits_gen.json` before running the model.


## Training

To train the boundary prediction module, run the following command in the `cg_boundary_prediction_module` directory:

```bash
$ python boundpred_train.py
```

Similarly, to train the crown generation module, run the following command in the `cg_generation_module` directory:

```bash
$ python gen_train.py
```

Please refer to the configuration files for recommended training parameters.

## Testing

To run inference with the boundary prediction module, execute the following command in the `cg_boundary_prediction_module` directory:

```bash
$ python boundpred_inference.py
```

Similarly, to run inference with the crown generation module, execute the following command in the `cg_generation_module` directory:

```bash
$ python gen_inference.py
```
To reconstruct watertight meshes from the generated point cloud crowns, ensure you have a trained DPSR model as described in [the Shape as Points repository](https://github.com/autonomousvision/shape_as_points). Configuration settings can be found in the `cg_generation_module/mesh_recon/configs` directory.

## Acknowledgements

This codebase is inspired by the following repositories. We greatly appreciate the authors for their contributions.
* https://github.com/autonomousvision/shape_as_points ([MIT license](https://github.com/autonomousvision/shape_as_points/blob/main/LICENSE))
* https://github.com/alexzhou907/PVD ([MIT license](https://github.com/alexzhou907/PVD/blob/main/LICENSE))
* https://github.com/plai-group/flexible-video-diffusion-modeling ([MIT license](https://github.com/plai-group/flexible-video-diffusion-modeling/blob/main/LICENSE))