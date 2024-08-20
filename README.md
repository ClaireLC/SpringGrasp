# SpringGrasp
## Synthesizing Compliant, Dexterous Grasp under Shape Uncertainty
Optimization based compliant grasp synthesis using only single depth image.

## Installation
### Install basic python dependencies
```
pip install -r requirements.txt
```

### Install thirdparty tools
```
cd thirdparty
cd differentiable-robot-model
pip install -e .
cd TorchSDF
bash install.sh
```
### Install curobo for arm motion planning and collision avoidance [optional]
```
mkdir curobo_ws
```
Download and unzip customized [curobo](https://drive.google.com/file/d/1uNE-5SKdsH63a3fXlR7KLqrvdTvE27bA/view?usp=sharing) inside `curobo_ws`

Follow install instruction of each package in their `README.md`.

## File structure
```
root directory
  ├── assets  
  │   └── // folders for real scanned object
  ├── data  
  │   └── // folders for data after optimization
  ├── gpis_states  
  │   └── // state data for restoring and visualizing gaussian process implicit surface
  ├── thirdparty  
  │   ├── // dependent third-party package
  |   └── TorchSDF
  |      └── // Compute SDF of mesh in pytorch
  |   └── differentiable-robot-model
  |      └── // Differentiable forward kinematics model
  ├── [curobo_ws] // customized curobo motion planner
  │   ├── curobo
  |   └── nvblox
  |   └── nvblox_torch
  ├── gpis
  |   ├── 3dplot.py // Visualizing GPIS intersection and its uncertainty
  |   └── gpis.py // Definition for Gaussian process implicit surface
  ├── spring_grasp_planner // Core implementation
  |   ├── initial_guesses.py // Initial wrist poses
  |   ├── metric.py // Implementation of spring grasp metric
  |   └── gpis.py // Different optimizers for spring grasp planner
  ├── utils
  |   └── // Ultilities to support the project
  ├── process_pcd.py // Processing pointclouds from different cameras
  ├── optimize_pregrasp.py // Running compliant pregrasp optimization
  └── verify_grasp_robot.py  // verifying pregrasp on hardware, kuka iiwa14 + left allegro hand.
```

## Usage (SpringGrasp + Evaluation Metrics)
### Plan grasp for pre-scanned objects
#### Original SpringGrasp
```
python optimize_pregrasp_.py --exp_name <obj_name>
```
```
python viz_utils_.py <file_name_here>
```

#### Modified SpringGrasp
The `optimize_pregrasp.py` and `viz_utils.py` has been modified. I recommend cloning the entire branch for smooth transition. I renamed the original `optimize_pregrasp.py` as `optimize_pregrasp_.py` and `viz_utils.py` as `viz_utils_.py`. In the `func_grasp_junho` branch, `optimize_pregrasp.py` and `viz_utils.py` are both modified versions.

1. Clone `func_grasp_junho` branch.
2. Run `python optimize_pregrasp.py`

### Customized object
#### Plan and visualize grasp
The objects are controlled through `OBJ_NAME` and `OBJ_NUM`. This will modify the `--npz_path ` argument within the file.  (line 192)

For now, all the objects collected and annotated through Affcorrs is in `juno/u/junhokim/code/zed_redis/pcd_data/`, and the `--npz_path` argument is set to this directory. Even though the objects are in my folder, `--npz_path` should correctly allow you to read the data.

The `ann_gt_pcd.npz` file is the annotated ground truth point cloud data which is generated from `e2_ann_corr_chamfer.py` in the `Affcorrs` directory. This is within the `/juno/u/junhokim/code/zed_redis/pcd_data/{OBJ_NAME}/{OBJ_NUM}/`.

If you want to run SpringGrasp on different categories and types of objects, modify the `OBJ_NAME` and `OBJ_NUM` in the  `optimize_pregrasp.py` file. (line 172~174)
```
python optimize_pregrasp.py
```
#### Saved Optimized Grasps
The save path is controlled by `--exp_name` argument (line 181). Currently it is set to `/juno/u/junhokim/code/SpringGrasp/data/grasp/{OBJ_NAME}/{OBJ_NUM}/`. The directory is controlled by `OBJ_NAME` and `OBJ_NUM`, so the parent directory should not matter as long as it is a valid path.

#### Visualize Hit or Miss & SpringGrasp vectors(Metric 3)
1. Change `GRASP_CALLER` to `True`
```
python optimiza_pregrasp.py
```
The modified `viz_utils.py` is called when `GRASP_CALLER` is set to `True`. It calls `viz_simple()` in line 308 to call `viz_utils.py` without re-running the entire optimization process again.

## Usage (original SpringGrasp)
### Plan grasp for pre-scanned objects
#### Process pointcloud
```
python process_pcd.py --exp_name <obj_name>
```
#### Plan and visualizing grasp
```
python optimize_pregrasp.py --exp_name <obj_name>
```
### Customized object
#### Plan and visualize grasp
```
python optimize_pregrasp.py --exp_name <obj_name> --pcd_file <path to your ply pointcloud>
```
#### Plan reaching motion for Kuka arm (require curobo installation and scene configuration)
```
python traj_gen.py --exp_name <obj_name>
```
#### Execute grasp on real robot(require support for kuka iiwa14 + left allegro hand)
```
python verify_grasp_robot.py --exp_name <obj_name>
```
### Using customized scene and deploy on hardware
How to deploy on hardware varies case by case, if you need help with using Kuka iiwa14 + allegro hand or run into troubles with coordinate system convention please contact: ericcsr [at] stanford [dot] edu