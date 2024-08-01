# RaDe-GS: Rasterizing Depth in Gaussian Splatting

### RaDe-GS: Rasterizing Depth in Gaussian Splatting
Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, Ping Tan

[Project page](https://baowenz.github.io/radegs/)
![Teaser image](assets/teaser.png)
# News! 
### 1. We have updated the formluation of RaDe-GS (as shown in the 'Modifications'). It achieves better performance on TNT dataset.
### 2. Now, we release the updated code of Marching Tetrahedra, based on [GOF](https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/eval_tnt/run.py)'s orginal proposal. In our version, opacities are calculated in ray space, which better fits our needs.

# Modifications

1. We change to calculate depth using per-pixel cosine values  (Eq. 14: $d=cos\theta\ t^*$). Additionally, an option is provided to directly render the camera space coordinate map. We utilize the inverse of affine approximation to transform intersections from ray space to camera space, and these transformed points are then used in computing normal consistency loss.
2. The depth distortion loss has been eliminated from our training process. Currently, we rely solely on normal consistency loss for geometry regularization.  We believe future techniques will enhance performance even further.


# 1. Installation
## Clone this repository.
```
git clone https://github.com/BaowenZ/RaDe-GS.git --recursive
```

## Install dependencies.
1. create an environment
```
conda create -n radegs python=3.9
conda activate radegs
```

2. install pytorch and other dependencies.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

3. install submodules
```
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

# tetra-nerf for Marching Tetrahedra
cd submodules/tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
# you can specify your own cuda path
# export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH
make 
pip install -e .
```

# 2. Preparation
We use preprocessed DTU dataset from [2DGS](https://surfsplatting.github.io/) for training. And we follow GOF to evaluate the geometry. Point clouds from the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36) need saved to dtu_eval/Offical_DTU_Dataset for the geometry evaluation.
We use preprocessed Tanks and Temples dataset from [GOF](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main). For evalution, please download ground truth point cloud, camera poses, alignments and cropfiles from [Tanks and Temples dataset](https://www.tanksandtemples.org/download/). The ground truth dataset should be organized as:
```
GT_TNT_dataset
│
└─── Barn
│   │
|   └─── Barn.json
│   │
|   └─── Barn.ply
│   │
|   └─── Barn_COLMAP_SfM.log
│   │
|   └─── Barn_trans.txt
│ 
└─── Caterpillar
│   │
......
```

# 3. Training and Evalution
## DTU Dataset
```
# training
python train.py -s <path to DTU dataset> -m <output folder> -r 2 --use_decoupled_appearance
# mesh extraction
python mesh_extract.py -s <path to DTU dataset> -m <output folder> -r 2
# evaluation
python evaluate_dtu_mesh.py -s <path to DTU dataset> -m <output folder>
```
## TNT Dataset
```
# training
python train.py -s <path to preprocessed TNT dataset> -m <output folder> -r 2 --eval --use_decoupled_appearance
# mesh extraction
python mesh_extract_tetrahedra.py -s <path to preprocessed TNT dataset> -m <output folder> -r 2 --eval
# evaluation
python eval_tnt/run.py --dataset-dir <path to GT TNT dataset> --traj-path <path to preprocessed TNT COLMAP_SfM.log file> --ply-path <output folder>/recon.ply
```
## Novel View Synthesis
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval
python render.py -m <path to pre-trained model> -s <path to dataset>
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```
Our model can directly render coordinate map for training, without the need to first render a depth map and then convert it. This feature can be activated by including `--use_coord_map` in the argument list of 'train.py'.

# 4. Viewer
Current viewer in this repository is very similar to the original Gaussian Splatting viewer (with small modifications for 3D filters).
You can build and use it in the same way as [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).


# 5. Acknowledge
We build this project based on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

We incorporate the filters proposed in [Mip-Splatting](https://github.com/autonomousvision/mip-splatting).

We incorporate the loss functions of [2D GS](https://github.com/hbb1/2d-gaussian-splatting) and use the preprocessed DTU dataset.

We incorporate the densification strategy, evalution and decoupled appearance modeling form [GOF](https://github.com/autonomousvision/gaussian-opacity-fields/tree/main)  and use the preprocessed TNT dataset.

The evaluation scripts for the DTU and Tanks and Temples datasets are sourced from [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation), respectively.

We thank the authors of Gaussian Splatting, Mip-Splatting, 2D GS, GOF， and the repos for their great works.