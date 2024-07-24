import os
import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import math
import numpy as np
from scene.cameras import Camera
from gaussian_renderer import render
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import depth_double_to_normal
import json


def load_camera(args):
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)




def extract_mesh(dataset, pipe, checkpoint_iterations=None):
    gaussians = GaussianModel(dataset.sh_degree)
    output_path = os.path.join(dataset.model_path,"point_cloud")
    iteration = 0
    if checkpoint_iterations is None:
        for folder_name in os.listdir(output_path):
            iteration= max(iteration,int(folder_name.split('_')[1]))
    else:
        iteration = checkpoint_iterations
    output_path = os.path.join(output_path,"iteration_"+str(iteration),"point_cloud.ply")

    gaussians.load_ply(output_path)
    print(f'Loaded gaussians from {output_path}')
    
    kernel_size = dataset.kernel_size
    
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_cam_list = load_camera(dataset)

    depth_list = []
    color_list = []
    alpha_thres = 0.5
    for viewpoint_cam in viewpoint_cam_list:
        # Rendering offscreen from that camera 
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
        # rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy()
        gt_image = viewpoint_cam.original_image.cuda()
        depth_middepth_normal = depth_double_to_normal(viewpoint_cam, render_pkg["expected_depth"], render_pkg["median_depth"])
        plt.figure()
        plt.subplot(3,2,1)
        plt.imshow(gt_image.clamp(0,1).detach().cpu().numpy().transpose(1,2,0))
        plt.subplot(3,2,2)
        plt.imshow(render_pkg["render"].clamp(0,1).detach().cpu().numpy().transpose(1,2,0))
        plt.subplot(3,2,3)
        plt.imshow(render_pkg["expected_depth"].detach().cpu().numpy().transpose(1,2,0)/10+0.5)
        plt.subplot(3,2,4)
        plt.imshow(render_pkg["median_depth"].detach().cpu().numpy().transpose(1,2,0)/10+0.5)
        plt.subplot(3,2,5)
        plt.imshow((-depth_middepth_normal[1].detach().cpu().numpy().transpose(1,2,0)+1)/2)
        plt.subplot(3,2,6)
        plt.imshow((-render_pkg["normal"].detach().cpu().numpy().transpose(1,2,0)+1)/2)
        plt.show()
        plt.close()

    print("done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)
        
        
    
    