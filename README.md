# RaDe-GS: Rasterizing Depth in Gaussian Splatting

### RaDe-GS: Rasterizing Depth in Gaussian Splatting
Baowen Zhang, Chuan Fang, Rakesh Shrestha, Yixun Liang, Xiaoxiao Long, Ping Tan

[Project page](https://baowenz.github.io/radegs/)
![Teaser image](assets/teaser.png)
### Thank you for your interest in our work! We have released the training and testing code on dtu. We are currently organizing the remaining code and will release it soon.

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
```

# 2. Preparation
We use preprocessed DTU dataset from [2DGS](https://surfsplatting.github.io/) for training. And we follow GOF to evaluate the geometry. Point clouds from the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36) need saved to dtu_eval/Offical_DTU_Dataset for the geometry evaluation. 

# 3. Training and Evalution
```
# training
python train.py -s <path to DTU dataset> -m <output folder> -r 2 --use_decoupled_appearance
# mesh extraction
python mesh_extract.py -s <path to DTU dataset> -m <output folder> -r 2
# evaluation
python evaluate_dtu_mesh.py -s <path to DTU dataset> -m <output folder>
```

# 4. Acknowledge
We build this project based on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

We adopt the loss functions of [2D GS](https://github.com/hbb1/2d-gaussian-splatting) and use the preprocessed DTU dataset.

We adopt the densification strategy, evalution and decoupled appearance modeling form [GOF](https://github.com/autonomousvision/gaussian-opacity-fields/tree/main).

We thank the authors of Gaussian Splatting, 2D GS, and GOF for their great works.