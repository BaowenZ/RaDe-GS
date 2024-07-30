/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float* view_points,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* depths,
		const float* ts,
		const float* camera_planes,
		const float2* ray_planes,
		const float* alphas,
		const float3* normals,
		const float* accum_coord,
		const float* accum_depth,
		const float* normal_length,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dpixel_coords,
		const float* dL_dpixel_mcoords,
		const float* dL_dpixel_depth,
		const float* dL_dpixel_mdepth,
		const float* dL_dalphas,
		const float* dL_dpixel_normals,
		const float* normalmap,
		const float focal_x, 
		const float focal_y,
		float3* dL_dmean3D,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_dts,
		float* dL_dcamera_planes,
		float2* dL_dray_planes,
		float* dL_dnormals,
		bool require_coord,
		bool require_depth);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const float kernel_size,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		const float3* dL_dview_points,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		const float* dL_dts,
		const float2* dL_dcamera_plane,
		const float2* dL_dray_plane,
		const float* dL_dnormals,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		const float4* conic_opacity,
		float* dL_dopacity);
}

#endif