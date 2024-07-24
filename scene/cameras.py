#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.gt_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.gt_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # the edge calculation is adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py
        with torch.no_grad():
            grad_img_left = torch.mean(torch.abs(self.original_image[:, 1:-1, 1:-1] - self.original_image[:, 1:-1, :-2]), 0)
            grad_img_right = torch.mean(torch.abs(self.original_image[:, 1:-1, 1:-1] - self.original_image[:, 1:-1, 2:]), 0)
            grad_img_top = torch.mean(torch.abs(self.original_image[:, 1:-1, 1:-1] - self.original_image[:, :-2, 1:-1]), 0)
            grad_img_bottom = torch.mean(torch.abs(self.original_image[:, 1:-1, 1:-1] - self.original_image[:, 2:, 1:-1]), 0)
            max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
            # pad
            max_grad = torch.exp(-max_grad)
            max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
            self.edge = max_grad



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

