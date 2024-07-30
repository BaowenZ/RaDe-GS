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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const float kernel_size,
			const bool prefiltered,
			float* out_color,
			float* out_coord,
			float* out_mcoord,
			float* out_depth,
			float* out_mdepth,
			float* out_alpha,
			float* out_normal,
			int* radii = nullptr,
			bool require_coord = true,
			bool require_depth = true,
			bool debug = false
			);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* alphas,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const float kernel_size,
			const int* radii,
			const float* normalmap,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dpix_coord,
			const float* dL_dpix_mcoord,
			const float* dL_dpix_depth,
			const float* dL_dpix_mdepth,
			const float* dL_dalphas,
			const float* dL_dpixel_normals,
			float* dL_dmean2D,
			float* dL_dview_points,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dts,
			float* dL_dcamera_planes,
			float* dL_dray_planes,
			float* dL_dnormals,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool require_coord = true,
			bool require_depth = true,
			bool debug = false);
		
		static int integrate(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> pointBuffer,
			std::function<char* (size_t)> point_binningBuffer,
			const int PN, const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* points3D,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* depths_plane_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const float kernel_size,
			const float* subpixel_offset,
			const bool prefiltered,
			float* out_color,
			float* accum_alpha,
			float* invraycov3Ds,
			int* radii = nullptr,
			float* out_alpha_integrated = nullptr,
			float* out_color_integrated = nullptr,
			float* out_coordinate2d = nullptr,
			float* out_sdf = nullptr,
			bool* condition = nullptr,
			bool debug = false);
	};
	
};

#endif