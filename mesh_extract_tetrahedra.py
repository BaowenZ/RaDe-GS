#adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra


@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            final_alpha = torch.min(final_alpha, alpha_integrated)
        alpha = 1 - final_alpha
    return alpha



@torch.no_grad()
def evaluage_cull_alpha(points, views, masks, gaussians, pipeline, background, kernel_size):
    # final_sdf = torch.zeros((points.shape[0]), dtype=torch.float32, device="cuda")
    final_sdf = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    weight = torch.zeros((points.shape[0]), dtype=torch.int32, device="cuda")
    with torch.no_grad():
        for cam_id, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.empty_cache()
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            point_coordinate = ret["point_coordinate"]
            point_coordinate[:,0] = (point_coordinate[:,0]*2+1)/(views[cam_id].image_width-1) - 1
            point_coordinate[:,1] = (point_coordinate[:,1]*2+1)/(views[cam_id].image_height-1) - 1
            rendered_mask = ret["render"][7]
            mask = rendered_mask[None]
            if not view.gt_mask is None:
                mask = mask * view.gt_mask
            if not masks is None:
                mask = mask * masks[cam_id]
            valid_point_prob = torch.nn.functional.grid_sample(mask.type(torch.float32)[None],point_coordinate[None,None],padding_mode='zeros',align_corners=False)
            valid_point_prob = valid_point_prob[0,0,0]
            valid_point = valid_point_prob>0.5
            final_sdf = torch.where(valid_point, torch.min(alpha_integrated,final_sdf), final_sdf)
            weight = torch.where(valid_point, weight+1, weight)
        final_sdf = torch.where(weight>0,0.5-final_sdf,-100)
    return final_sdf

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians: GaussianModel, pipeline, background, kernel_size):
    
    # generate tetra points here
    points, points_scale = gaussians.get_tetra_points()
    cells = cpp.triangulate(points)

    mask = None
    sdf = evaluage_cull_alpha(points, views, mask, gaussians, pipeline, background, kernel_size)

    torch.cuda.empty_cache()
    # the function marching_tetrahedra costs much memory, so we move it to cpu.
    verts_list, scale_list, faces_list, _ = marching_tetrahedra(points.cpu()[None], cells.cpu().long(), sdf[None].cpu(), points_scale[None].cpu())
    del points
    del points_scale
    del cells
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    end_points, end_sdf, end_scales = end_points.cuda(), end_sdf.cuda(), end_scales.cuda()
    
    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale
    
    n_binary_steps = 8
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        mid_sdf = evaluage_cull_alpha(mid_points, views, mask, gaussians, pipeline, background, kernel_size)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        points = (left_points + right_points) / 2

        
    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
    # filter
    vertice_mask = (distance <= scale).cpu().numpy()
    face_mask = vertice_mask[faces].all(axis=1)
    mesh.update_vertices(vertice_mask)
    mesh.update_faces(face_mask)

    mesh.export(os.path.join(model_path,"recon.ply"))

    

def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        cams = scene.getTrainCameras()
        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args))