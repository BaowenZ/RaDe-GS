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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ncc(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    sigma1 = torch.sqrt(sigma1_sq + 1e-4)
    sigma2 = torch.sqrt(sigma2_sq + 1e-4)

    image1_norm = (img1 - mu1) / (sigma1 + 1e-8)
    image2_norm = (img2 - mu2) / (sigma2 + 1e-8)

    ncc = F.conv2d((image1_norm * image2_norm), window, padding=0, groups=channel)

    return torch.mean(ncc, dim=2)

    
# def _ncc(pred, gt, window, channel):
#     ntotpx, nviews, nc, h, w = pred.shape
#     flat_pred = pred.view(-1, nc, h, w)
#     mu1 = F.conv2d(flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc)
#     mu2 = F.conv2d(gt, window, padding=0, groups=channel).view(ntotpx, nc)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2).unsqueeze(1)  # (ntotpx, 1, nc)

#     sigma1_sq = F.conv2d(flat_pred * flat_pred, window, padding=0, groups=channel).view(ntotpx, nviews, nc) - mu1_sq
#     sigma2_sq = F.conv2d(gt * gt, window, padding=0, groups=channel).view(ntotpx, 1, 3) - mu2_sq

#     sigma1 = torch.sqrt(sigma1_sq + 1e-4)
#     sigma2 = torch.sqrt(sigma2_sq + 1e-4)

#     pred_norm = (pred - mu1[:, :, :, None, None]) / (sigma1[:, :, :, None, None] + 1e-8)  # [ntotpx, nviews, nc, h, w]
#     gt_norm = (gt[:, None, :, :, :] - mu2[:, None, :, None, None]) / (
#             sigma2[:, :, :, None, None] + 1e-8)  # ntotpx, nc, h, w

#     ncc = F.conv2d((pred_norm * gt_norm).view(-1, nc, h, w), window, padding=0, groups=channel).view(
#         ntotpx, nviews, nc)

#     return torch.mean(ncc, dim=2)

