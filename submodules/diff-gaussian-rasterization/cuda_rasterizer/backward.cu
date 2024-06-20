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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	const float2* dL_depths_plane,
	const float3* dL_dnormals,
	float3* dL_dmeans,
	float* dL_dcov,
	const float4* __restrict__ conic_opacity,
	float* dL_dopacity)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	const glm::vec3 dL_dnormal = { dL_dnormals[idx].x,dL_dnormals[idx].y,dL_dnormals[idx].z};
	const float4 conic = conic_opacity[idx];
	const float combined_opacity = conic.w;
	const float2& dL_depth_plane_combined = dL_depths_plane[idx];
	// const float2& dL_depth_plane = dL_depths_plane[idx];
	// printf("%.10f %.10f \n",dL_depth_plane.x,dL_depth_plane.y);
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	float txtz = t.x / t.z;
	float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	txtz = t.x / t.z;
	tytz = t.y / t.z;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	const float det_0 = max(1e-6, cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1]);
	const float det_1 = max(1e-6, (cov2D[0][0] + kernel_size) * (cov2D[1][1] + kernel_size) - cov2D[0][1] * cov2D[0][1]);
	// sqrt here
	const float coef = sqrt(det_0 / (det_1+1e-6) + 1e-6);
	// const float coef = 1.0f;

	glm::mat3 Vrk_eigen_vector;
	glm::vec3 Vrk_eigen_value;
	int D = glm::findEigenvaluesSymReal(Vrk,Vrk_eigen_value,Vrk_eigen_vector);

	unsigned int min_id = Vrk_eigen_value[0]>Vrk_eigen_value[1]? (Vrk_eigen_value[1]>Vrk_eigen_value[2]?2:1):(Vrk_eigen_value[0]>Vrk_eigen_value[2]?2:0);

	glm::mat3 Vrk_inv;
	glm::vec3 eigenvector_min;
	bool well_conditioned = Vrk_eigen_value[min_id]>0.00000001;
	if(well_conditioned)
	{
		glm::mat3 diag = glm::mat3(1/Vrk_eigen_value[0],0,0,
									0,1/Vrk_eigen_value[1],0,
									0,0,1/Vrk_eigen_value[2]);
		Vrk_inv = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
	}
	else
	{
		if(D<3)
		{
			const glm::vec3 eigenvector1 = Vrk_eigen_vector[(min_id+1)%3];
			const glm::vec3 eigenvector2 = Vrk_eigen_vector[(min_id+2)%3];
			eigenvector_min = glm::cross(eigenvector1, eigenvector2);
		}
		else{
			eigenvector_min = Vrk_eigen_vector[min_id];
		}
		Vrk_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
	}
	
	// glm::mat3 Vrk_inv = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
	glm::mat3 cov_cam_inv = glm::transpose(W) * Vrk_inv * W;
	glm::vec3 uvh = {txtz, tytz, 1};
	glm::vec3 uvh_m = cov_cam_inv * uvh;
	glm::vec3 uvh_mn = glm::normalize(uvh_m);
	
	float vb = glm::dot(uvh_m, uvh);
	float vbn = glm::dot(uvh_mn, uvh);
	float u2 = txtz * txtz;
	float v2 = tytz * tytz;
	float uv = txtz * tytz;

	float l = sqrt(t.x*t.x+t.y*t.y+t.z*t.z);
	glm::mat3 Jn = glm::mat3(
		1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
		0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
		t.x/l, t.y/l, t.z/l);

	glm::mat3 nJ_inv = glm::mat3(
		v2 + 1,	-uv, 		0,
		-uv,	u2 + 1,		0,
		-txtz,	-tytz,		0
	);

	float clamp_vb = max(vb, 0.0000001f);
	float clamp_vbn = max(vbn, 0.0000001f);
	float factor = t.z / ((u2+v2+1) * clamp_vb);
	float factor2 = t.z / (u2+v2+1);
	float factor_normal = l / (u2+v2+1);
	glm::vec3 uvh_m_vb = uvh_mn/clamp_vbn;
	glm::vec3 planen = nJ_inv * uvh_m_vb;
	float2 depth_plane = { planen[0]*factor2, planen[1]*factor2};

	glm::vec3 ray_normal_vector = {-planen[0]*factor_normal, -planen[1]*factor_normal, -1};

	glm::vec3 cam_normal_vector = Jn * ray_normal_vector;
	glm::vec3 normal_vector = glm::normalize(cam_normal_vector);
	float lv2 = (cam_normal_vector.x*cam_normal_vector.x+\
				cam_normal_vector.y*cam_normal_vector.y+\
				cam_normal_vector.z*cam_normal_vector.z);
	float lv = sqrt(lv2);
	float lv3 = lv2 * lv;
	glm::mat3 eye3 = glm::mat3(1,0,0,
								0,1,0,
								0,0,1);
	const glm::vec3 dL_dnormal_lv = dL_dnormal/lv;
	glm::vec3 dL_dcam_normal_vector = dL_dnormal_lv - normal_vector * glm::dot(normal_vector,dL_dnormal_lv);
	glm::vec3 dL_dray_normal_vector = glm::transpose(Jn) * dL_dcam_normal_vector;
	const float l_z = l / t.z;
	const float dL_dl_z = - (dL_dray_normal_vector.x * depth_plane.x + dL_dray_normal_vector.y * depth_plane.y);
	const glm::vec2 dL_depth_plane = {dL_depth_plane_combined.x - dL_dray_normal_vector.x * l_z, dL_depth_plane_combined.y - dL_dray_normal_vector.y * l_z};
	glm::mat3 dL_dJn = glm::outerProduct(dL_dcam_normal_vector,ray_normal_vector);

	float tmp = dL_depth_plane.x * depth_plane.x + dL_depth_plane.y * depth_plane.y;
	float dL_dvb = -tmp / clamp_vb;

	glm::vec3 dL_dplane = {dL_depth_plane.x * (factor), dL_depth_plane.y * (factor), 0};

	

	glm::vec3 W_uvh = W * uvh;
	glm::vec3 nJ_inv_dL_dplane = glm::transpose(nJ_inv) * dL_dplane;
	glm::mat3 dL_dVrk = glm::mat3();

	if(well_conditioned){
		dL_dVrk = - glm::outerProduct(Vrk_inv * W_uvh, (Vrk_inv/clamp_vb) * (W_uvh * (-tmp) + W * glm::transpose(nJ_inv) * glm::vec3(dL_depth_plane.x * factor2, dL_depth_plane.y * factor2, 0)));
	}
	else{
		dL_dVrk = glm::mat3(0,0,0,0,0,0,0,0,0);
		glm::mat3 dL_dVrk_inv = glm::outerProduct(W_uvh, W_uvh * dL_dvb + W * nJ_inv_dL_dplane);
		glm::vec3 dL_dv = (dL_dVrk_inv + glm::transpose(dL_dVrk_inv)) * eigenvector_min;;
		for(int j =0;j<3;j++)
		{
			if(j!=min_id)
			{
				float scale = glm::dot(Vrk_eigen_vector[j],dL_dv)/min(Vrk_eigen_value[min_id] - Vrk_eigen_value[j], - 0.0000001f);
				dL_dVrk += glm::outerProduct(Vrk_eigen_vector[j] * scale, eigenvector_min);
			}
		}
	}
	
	
	glm::vec3 vec_dL_depth_plane = glm::vec3(dL_depth_plane.x,dL_depth_plane.y,0);
	glm::vec3 dL_duvh = 2 * (-tmp) * uvh_m_vb + (cov_cam_inv/clamp_vb) * glm::transpose(nJ_inv) * glm::vec3(dL_depth_plane.x * factor2, dL_depth_plane.y * factor2, 0);
	
	glm::mat3 dL_dnJ_inv = glm::outerProduct(vec_dL_depth_plane, uvh_m_vb * factor2);

	float dL_du = -tmp * 2 * txtz / (u2+v2+1)
					+ dL_duvh.x 
					+ (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-tytz) + 2 * dL_dnJ_inv[1][1] * txtz - dL_dnJ_inv[2][0]
					+ dL_dl_z * txtz / sqrt(u2+v2+1); //this line is from normal
	float dL_dv = -tmp * 2 * tytz / (u2+v2+1)
					+ dL_duvh.y 
					+ (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-txtz) + 2 * dL_dnJ_inv[0][0] * tytz - dL_dnJ_inv[2][1]
					+ dL_dl_z * tytz / sqrt(u2+v2+1); //this line is from normal


	const float opacity = combined_opacity / (coef + 1e-6);
	const float dL_dcoef = dL_dopacity[idx] * opacity;
	const float dL_dsqrtcoef = dL_dcoef * 0.5 * 1. / (coef + 1e-6);
	const float dL_ddet0 = dL_dsqrtcoef / (det_1+1e-6);
	const float dL_ddet1 = dL_dsqrtcoef * det_0 * (-1.f / (det_1 * det_1 + 1e-6));
	//TODO gradient is zero if det_0 or det_1 < 0
	const float dcoef_da = dL_ddet0 * cov2D[1][1] + dL_ddet1 * (cov2D[1][1] + kernel_size);
	const float dcoef_db = dL_ddet0 * (-2. * cov2D[0][1]) + dL_ddet1 * (-2. * cov2D[0][1]);
	const float dcoef_dc = dL_ddet0 * cov2D[0][0] + dL_ddet1 * (cov2D[0][0] + kernel_size);
	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] + kernel_size;
	float b = cov2D[0][1];
	float c = cov2D[1][1] + kernel_size;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		if (det_0 <= 1e-6 || det_1 <= 1e-6){
			dL_dopacity[idx] = 0;
		} else {
			// Gradiends of alpha respect to conv due to low pass filter
			dL_da += dcoef_da;
			dL_dc += dcoef_dc;
			dL_db += dcoef_db;

			// update dL_dopacity
			dL_dopacity[idx] = dL_dopacity[idx] * coef;
		}

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}
	dL_dcov[6 * idx + 0] += dL_dVrk[0][0];
	dL_dcov[6 * idx + 3] += dL_dVrk[1][1];
	dL_dcov[6 * idx + 5] += dL_dVrk[2][2];
	dL_dcov[6 * idx + 1] += dL_dVrk[0][1] + dL_dVrk[1][0];
	dL_dcov[6 * idx + 2] += dL_dVrk[0][2] + dL_dVrk[2][0];
	dL_dcov[6 * idx + 4] += dL_dVrk[1][2] + dL_dVrk[2][1];


	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;


	float dL_dtx = x_grad_mul * (-h_x * tz2 * dL_dJ02 + dL_du * tz
								-dL_dJn[0][2]*tz2 + dL_dJn[2][0]*(1/lv-t.x*t.x/lv3) + dL_dJn[2][2]*(-t.x*t.z/lv3)); //this line is from normal
	float dL_dty = y_grad_mul * (-h_y * tz2 * dL_dJ12 + dL_dv * tz
								-dL_dJn[1][2]*tz2 + dL_dJn[2][1]*(1/lv-t.y*t.y/lv3) + dL_dJn[2][2]*(-t.y*t.z/lv3)); //this line is from normal
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12
					- (dL_du * t.x + dL_dv * t.y) * tz2 + tmp * tz
					+ dL_dJn[0][0] * (-tz2) + dL_dJn[1][1] * (-tz2) + dL_dJn[0][2] * (2*t.x*tz3) + dL_dJn[1][2] * (2*t.y*tz3)
					+ (dL_dJn[2][0]*t.x+dL_dJn[2][1]*t.y)*(-t.z/lv3) + dL_dJn[2][2]*(1/lv-t.z*t.z/lv3); // two lines are from normal


	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	const float* dL_ddepth,
	const float2* dL_ddepth_plane,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// the w must be equal to 1 for view^T * [x,y,z,1]
	float3 m_view = transformPoint4x3(m, view);

	// Compute loss gradient w.r.t. 3D means due to gradients of depth
	// from rendering procedure
	glm::vec3 dL_dmean2;
	float mul3 = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
	dL_dmean2.x = (view[2] - view[3] * mul3) * dL_ddepth[idx];
	dL_dmean2.y = (view[6] - view[7] * mul3) * dL_ddepth[idx];
	dL_dmean2.z = (view[10] - view[11] * mul3) * dL_ddepth[idx];

	// That's the third part of the mean gradient.
	dL_dmeans[idx] += dL_dmean2;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float2* __restrict__ depths_plane,
	const float* __restrict__ alphas,
	const float3* __restrict__ normals,
	const float* __restrict__ wd_map,
	const float* __restrict__ wd2_map,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_depths,
	const float* __restrict__ dL_dpixel_middepths,
	const float* __restrict__ dL_dalphas,
	const float* __restrict__ dL_dpixel_normals,
	const float* __restrict__ dL_ddistortions,
	const float focal_x, 
	const float focal_y,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths,
	float2* __restrict__ dL_ddepths_plane,
	float3* __restrict__ dL_dnormals
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float2 collected_plane[BLOCK_SIZE];
	__shared__ float collected_normals[3 * BLOCK_SIZE];


	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - alphas[pix_id]) : 0;
	const float w_final = inside ? alphas[pix_id] : 0;
	const float wd_final = inside ? wd_map[pix_id] : 0;
	const float wd2_final = inside ? wd2_map[pix_id] : 0;
	
	float T = T_final;
	float w = w_final;
	float wd = wd_final;
	float wd2 = wd2_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	const int max_contributor = inside ? n_contrib[pix_id + H * W] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float accum_depth_rec = 0;
	float dL_ddistortion;
	float accum_distortion_rec = 0;
	float dL_dpixel_depth;
	float accum_alpha_rec = 0;
	float dL_dalpha;
	float accum_normal_rec[3] = {0};
	float dL_dpixel_normal[3];
	float dL_dpixel_middepth=0;
	float accum_w = 0;
	float accum_wd = 0;
	float accum_wd2 = 0;
	
	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];
		dL_dpixel_middepth = dL_dpixel_middepths[pix_id];
		dL_dalpha = dL_dalphas[pix_id];
		dL_ddistortion = dL_ddistortions[pix_id];
		for (int i = 0; i < 3; i++)
			dL_dpixel_normal[i] = dL_dpixel_normals[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	float last_dL_dw = 0;
	float last_normal[3] = {0};

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
			float2 depth_plane = depths_plane[coll_id]; 
			collected_plane[block.thread_rank()] = {depth_plane.x/focal_x,depth_plane.y/focal_y};
			{
				collected_normals[0 * BLOCK_SIZE + block.thread_rank()] = normals[coll_id].x;
				collected_normals[1 * BLOCK_SIZE + block.thread_rank()] = normals[coll_id].y;
				collected_normals[2 * BLOCK_SIZE + block.thread_rank()] = normals[coll_id].z;
			}
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];

			const float c_d = collected_depths[j];
			const float2 depth_plane = collected_plane[j];
			float depth = c_d + (depth_plane.x * d.x + depth_plane.y * d.y);
			if (depth <= NEAR_PLANE)
				continue;

			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f){
				continue;
			}

			

			const float mapped_max_t = (FAR_PLANE * depth - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth);
			const float dmax_t_dd = (FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth * depth);

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			// const float alpha = con_o.w * G;
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float& dpixel_depth_ddepth = dchannel_dcolor;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dopa = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dopa += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			for (int ch = 0; ch < 3; ch++)
			{
				const float c = collected_normals[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = c;

				const float dL_dchannel = dL_dpixel_normal[ch];
				dL_dopa += (c - accum_normal_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				switch (ch)
				{
				case 0:
					atomicAdd(&dL_dnormals[global_id].x, dchannel_dcolor * dL_dchannel);
					break;
				case 1:
					atomicAdd(&dL_dnormals[global_id].y, dchannel_dcolor * dL_dchannel);
					break;
				case 2:
					atomicAdd(&dL_dnormals[global_id].z, dchannel_dcolor * dL_dchannel);
					break;
				default:
					break;
				}
			}
			
			// Propagate gradients from pixel depth to opacity

			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = depth;
			dL_dopa += (depth - accum_depth_rec) * dL_dpixel_depth;
			float dL_ddepth = dpixel_depth_ddepth * dL_dpixel_depth
								+ 2 * dchannel_dcolor * (w_final * mapped_max_t - wd_final) * dL_ddistortion * dmax_t_dd;
			if (contributor == max_contributor-1) {
				dL_ddepth += dL_dpixel_middepth;
			}
			
			atomicAdd(&(dL_ddepths[global_id]), dL_ddepth);
			atomicAdd(&dL_ddepths_plane[global_id].x, dL_ddepth * d.x / focal_x);
			atomicAdd(&dL_ddepths_plane[global_id].y, dL_ddepth * d.y / focal_y);

			// Propagate gradients from pixel alpha (weights_sum) to opacity
			accum_alpha_rec = last_alpha + (1.f - last_alpha) * accum_alpha_rec;
			dL_dopa += (1 - accum_alpha_rec) * dL_dalpha; //- (alpha - accum_alpha_rec) * dL_dalpha;

			
			// const float d2 = mapped_max_t * mapped_max_t;
			// const float& weight = dchannel_dcolor;
			// w -= weight;
			// wd -= weight * mapped_max_t;
			// wd2 -= weight * d2;
			// float dL1_dw = w * d2 + accum_wd2;
			// float dL2_dw = wd2 + accum_w * d2;
			// float dL3_dw = (wd+accum_wd)*(-2*mapped_max_t);
			// float dL_dw = dL1_dw + dL2_dw + dL3_dw;
			// accum_distortion_rec = last_alpha * last_dL_dw + (1.f - last_alpha) * accum_distortion_rec;
			// dL_dopa += (dL_dw - accum_distortion_rec) * dL_ddistortion;
			// last_dL_dw = dL_dw;

			// accum_w += weight;
			// accum_wd += weight * mapped_max_t;
			// accum_wd2 += weight * d2;



			dL_dopa *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dopa += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dopa;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, (dL_dG * dG_ddelx + dL_ddepth * depth_plane.x) * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, (dL_dG * dG_ddely + dL_ddepth * depth_plane.y) * ddely_dy);
			// fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields
			const float abs_dL_dmean2D = abs(dL_dG * dG_ddelx * ddelx_dx) + abs(dL_dG * dG_ddely * ddely_dy);
            atomicAdd(&dL_dmean2D[global_id].z, abs_dL_dmean2D);
			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dopa);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	const float* dL_ddepth,
	const float2* dL_ddepth_plane,
	const float3* dL_dnormals,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	const float4* conic_opacity,
	float* dL_dopacity)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		dL_ddepth_plane,
		dL_dnormals,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		conic_opacity,
		dL_dopacity);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_ddepth_plane,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float2* depths_plane,
	const float* alphas,
	const float3* normals,
	const float* wd_map,
	const float* wd2_map,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dpixel_depths,
	const float* dL_dpixel_middepths,
	const float* dL_dalphas,
	const float* dL_dpixel_normals,
	const float* dL_ddistortions,
	const float focal_x, 
	const float focal_y,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths,
	float2* dL_ddepths_plane,
	float3* dL_dnormals)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		depths_plane,
		alphas,
		normals,
		wd_map,
		wd2_map,
		n_contrib,
		dL_dpixels,
		dL_dpixel_depths,
		dL_dpixel_middepths,
		dL_dalphas,
		dL_dpixel_normals,
		dL_ddistortions,
		focal_x, 
		focal_y,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths,
		dL_ddepths_plane,
		dL_dnormals
		);
}