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
	const float kernel_size,
	const float* view_matrix,
	const float* dL_dconics,
	const float2* dL_dcamera_planes,
	const float2* dL_dray_planes,
	const float* dL_dnormals,
	glm::vec3* dL_dmeans,
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
	const glm::vec3 dL_dnormal = { dL_dnormals[idx*3],dL_dnormals[idx*3+1],dL_dnormals[idx*3+2]};
	const float4 conic = conic_opacity[idx];
	const float combined_opacity = conic.w;
	const float2 dL_camera_plane0 = dL_dcamera_planes[idx*3];
	const float2 dL_camera_plane1 = dL_dcamera_planes[idx*3+1];
	const float2 dL_camera_plane2 = dL_dcamera_planes[idx*3+2];

	const float2 dL_dray_plane = dL_dray_planes[idx];

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
	int D = glm_modification::findEigenvaluesSymReal(Vrk,Vrk_eigen_value,Vrk_eigen_vector);

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
		eigenvector_min = Vrk_eigen_vector[min_id];
		Vrk_inv = glm::outerProduct(eigenvector_min,eigenvector_min);
	}
	
	// glm::mat3 Vrk_inv = Vrk_eigen_vector * diag * glm::transpose(Vrk_eigen_vector);
	glm::mat3 cov_cam_inv = glm::transpose(W) * Vrk_inv * W;
	glm::vec3 uvh = {txtz, tytz, 1};
	glm::vec3 uvh_m = cov_cam_inv * uvh;
	glm::vec3 uvh_mn = glm::normalize(uvh_m);


	float u2 = txtz * txtz;
	float v2 = tytz * tytz;
	float uv = txtz * tytz;

	glm::mat3 dL_dVrk;
	glm::vec3 plane;
	float dL_du;
	float dL_dv;
	float dL_dl;
	float l;
	float nl;
	glm::mat3 dL_dnJ;
	if(isnan(uvh_mn.x)||D==0)
	{
		dL_dVrk = glm::mat3(0,0,0,0,0,0,0,0,0);
		dL_dnJ = glm::mat3(0,0,0,0,0,0,0,0,0);
		plane = glm::vec3(0,0,0);
		nl = 1;
		l = 1;
		dL_du = 0;
		dL_dv = 0;
		dL_dl = 0;
	}
	else
	{
		float vb = glm::dot(uvh_m, uvh);
		float vbn = glm::dot(uvh_mn, uvh);

		l = sqrt(t.x*t.x+t.y*t.y+t.z*t.z);
		glm::mat3 nJ = glm::mat3(
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
		nl = u2+v2+1;
		float factor_normal = l / nl;
		glm::vec3 uvh_m_vb = uvh_mn/clamp_vbn;
		plane = nJ_inv * uvh_m_vb;
		
		glm::vec2 camera_plane0 = {(-(v2 + 1)*t.z+plane[0]*t.x)/nl, (uv*t.z+plane[1]*t.x)/nl};
		glm::vec2 camera_plane1 = {(uv*t.z+plane[0]*t.y)/nl, (-(u2 + 1)*t.z+plane[1]*t.y)/nl};
		glm::vec2 camera_plane2 = {(t.x+plane[0]*t.z)/nl, (t.y+plane[1]*t.z)/nl};

		glm::vec2 ray_plane = {plane[0]*factor_normal, plane[1]*factor_normal};
		glm::vec3 ray_normal_vector = {-plane[0]*factor_normal, -plane[1]*factor_normal, -1};

		glm::vec3 cam_normal_vector = nJ * ray_normal_vector;
		glm::vec3 normal_vector = glm::normalize(cam_normal_vector);
		float lv = glm::length(cam_normal_vector);
		const glm::vec3 dL_dnormal_lv = dL_dnormal/lv;
		glm::vec3 dL_dcam_normal_vector = dL_dnormal_lv - normal_vector * glm::dot(normal_vector,dL_dnormal_lv);
		glm::vec3 dL_dray_normal_vector = glm::transpose(nJ) * dL_dcam_normal_vector;
		dL_dnJ = glm::outerProduct(dL_dcam_normal_vector,ray_normal_vector);
		dL_dl = (-plane[0] * dL_dray_normal_vector.x - plane[1] * dL_dray_normal_vector.y
					+ plane[0] * dL_dray_plane.x + plane[1] * dL_dray_plane.y) / nl;
		
		glm::vec2 dL_dplane = glm::vec2(
			(t.x*dL_camera_plane0.x + t.y*dL_camera_plane1.x + t.z*dL_camera_plane2.x 
												 -l * dL_dray_normal_vector[0] + dL_dray_plane.x * l) / nl,
			(t.x*dL_camera_plane0.y + t.y*dL_camera_plane1.y + t.z*dL_camera_plane2.y
												 -l * dL_dray_normal_vector[1] + dL_dray_plane.y * l) / nl
		);
		glm::vec3 dL_dplane_append = glm::vec3(dL_dplane.x, dL_dplane.y, 0);

		float dL_dnl = (-dL_camera_plane0.x * camera_plane0.x - dL_camera_plane0.y * camera_plane0.y
						-dL_camera_plane1.x * camera_plane1.x - dL_camera_plane1.y * camera_plane1.y
						-dL_camera_plane2.x * camera_plane2.x - dL_camera_plane2.y * camera_plane2.y
						-dL_dray_normal_vector[0] * ray_normal_vector.x - dL_dray_normal_vector[1] * ray_normal_vector.y
						-dL_dray_plane.x * ray_plane.x - dL_dray_plane.y * ray_plane.y) / nl;


		float tmp = dL_dplane.x * plane.x + dL_dplane.y * plane.y;

		glm::vec3 W_uvh = W * uvh;

		if(well_conditioned){
			dL_dVrk = - glm::outerProduct(Vrk_inv * W_uvh, (Vrk_inv/clamp_vb) * (W_uvh * (-tmp) + W * glm::transpose(nJ_inv) * dL_dplane_append));
		}
		else{
			dL_dVrk = glm::mat3(0,0,0,0,0,0,0,0,0);
			float dL_dvb = -tmp / clamp_vb;
			glm::vec3 nJ_inv_dL_dplane = glm::transpose(nJ_inv) * glm::vec3(dL_dplane.x / clamp_vb, dL_dplane.y / clamp_vb, 0);
			glm::mat3 dL_dVrk_inv = glm::outerProduct(W_uvh, W_uvh * dL_dvb + W * nJ_inv_dL_dplane);
			glm::vec3 dL_dv = (dL_dVrk_inv + glm::transpose(dL_dVrk_inv)) * eigenvector_min;
			for(int j =0;j<3;j++)
			{
				if(j!=min_id)
				{
					float scale = glm::dot(Vrk_eigen_vector[j],dL_dv)/min(Vrk_eigen_value[min_id] - Vrk_eigen_value[j], - 0.0000001f);
					dL_dVrk += glm::outerProduct(Vrk_eigen_vector[j] * scale, eigenvector_min);
				}
			}
		}
		
		
		glm::vec3 dL_duvh = 2 * (-tmp) * uvh_m_vb + (cov_cam_inv/clamp_vb) * glm::transpose(nJ_inv) * dL_dplane_append;
		
		glm::mat3 dL_dnJ_inv = glm::outerProduct(dL_dplane_append, uvh_m_vb);
		
		dL_du = dL_dnl * 2 * txtz
				+ dL_duvh.x 
				+ (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-tytz) + 2 * dL_dnJ_inv[1][1] * txtz - dL_dnJ_inv[2][0]
				+ (dL_camera_plane0.y * t.y + dL_camera_plane1.x * t.y + dL_camera_plane1.y * (-2*t.x)) / nl;
		dL_dv = dL_dnl * 2 * tytz
				+ dL_duvh.y 
				+ (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-txtz) + 2 * dL_dnJ_inv[0][0] * tytz - dL_dnJ_inv[2][1]
				+ (dL_camera_plane0.x * (-2*t.y) + dL_camera_plane0.y * t.x + dL_camera_plane1.x * t.x) / nl;;
	}

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

	// glm::mat3 nJ = glm::mat3(
	// 		1 / t.z, 0.0f, -(t.x) / (t.z * t.z),
	// 		0.0f, 1 / t.z, -(t.y) / (t.z * t.z),
	// 		t.x/l, t.y/l, t.z/l);
	float l3 = l * l * l;
	float dL_dtx = x_grad_mul * (-h_x * tz2 * dL_dJ02 + dL_du * tz
								-dL_dnJ[0][2]*tz2 + dL_dnJ[2][0]*(1/l-t.x*t.x/l3) + dL_dnJ[2][1]*(-t.x*t.y/l3) + dL_dnJ[2][2]*(-t.x*t.z/l3) //this line is from normal
								+(dL_camera_plane0.x * plane[0] + dL_camera_plane0.y * plane[1] + dL_camera_plane2.x)/nl
								+dL_dl*t.x/l);
	float dL_dty = y_grad_mul * (-h_y * tz2 * dL_dJ12 + dL_dv * tz
								-dL_dnJ[1][2]*tz2 + dL_dnJ[2][0]*(-t.x*t.y/l3) + dL_dnJ[2][1]*(1/l-t.y*t.y/l3) + dL_dnJ[2][2]*(-t.y*t.z/l3) //this line is from normal
								+(dL_camera_plane1.x * plane[0] + dL_camera_plane1.y * plane[1] + dL_camera_plane2.y)/nl
								+dL_dl*t.y/l);
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12
					- (dL_du * t.x + dL_dv * t.y) * tz2
					+ (dL_dnJ[0][0] + dL_dnJ[1][1]) * (-tz2) + dL_dnJ[0][2] * (2*t.x*tz3) + dL_dnJ[1][2] * (2*t.y*tz3) // two lines are from normal
					+ (dL_dnJ[2][0]*t.x+dL_dnJ[2][1]*t.y)*(-t.z/l3) + dL_dnJ[2][2]*(1/l-t.z*t.z/l3) // two lines are from normal
					+ (dL_camera_plane0.x * (-(v2 + 1)) + dL_camera_plane0.y * uv + dL_camera_plane1.x * uv + dL_camera_plane1.y * (-(u2+1)) + dL_camera_plane2.x * plane[0] + dL_camera_plane2.y * plane[1])/nl
					+ dL_dl*t.z/l;


	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
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
	const float3* dL_dview_points,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	const float* dL_dts,
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
	glm::vec3 dL_dmean1;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean1.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean1.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean1.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// the w must be equal to 1 for view^T * [x,y,z,1]
	float3 m_view = transformPoint4x3(m, view);

	float t = sqrt(m_view.x*m_view.x+m_view.y*m_view.y+m_view.z*m_view.z);
	float dL_dt = dL_dts[idx];

	float3 dL_dview_point = dL_dview_points[idx];

	float3 dL_dmean2 = transformVec4x3Transpose({dL_dview_point.x+m_view.x/t*dL_dt,
												dL_dview_point.y+m_view.y/t*dL_dt,
												dL_dview_point.z+m_view.z/t*dL_dt}, view);

	// That's the third part of the mean gradient.
	dL_dmeans[idx] += glm::vec3(
		dL_dmean1.x + dL_dmean2.x,
		dL_dmean1.y + dL_dmean2.y,
		dL_dmean1.z + dL_dmean2.z
	);

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C, bool COORD = true, bool DEPTH = true, bool NORMAL = true>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float* __restrict__ view_points,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ ts,
	const float* __restrict__ camera_planes,
	const float2* __restrict__ ray_planes,
	const float* __restrict__ alphas,
	const float3* __restrict__ normals,
	const float* __restrict__ accum_coord,
	const float* __restrict__ accum_depth,
	const float* __restrict__ normal_length,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dpixel_coords,
	const float* __restrict__ dL_dpixel_mcoords,
	const float* __restrict__ dL_dpixel_depths,
	const float* __restrict__ dL_dpixel_mdepths,
	const float* __restrict__ dL_dalphas,
	const float* __restrict__ dL_dpixel_normals,
	const float* __restrict__ normalmap,
	const float focal_x, 
	const float focal_y,
	float3* __restrict__ dL_dmean3D,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dts,
	float* __restrict__ dL_dcamera_planes,
	float2* __restrict__ dL_dray_planes,
	float* __restrict__ dL_dnormals
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
	const float2 pixnf = {(pixf.x-W/2.f)/focal_x,(pixf.y-H/2.f)/focal_y};
	const float ln = sqrt(pixnf.x*pixnf.x+pixnf.y*pixnf.y+1);

	constexpr bool GEO = COORD || DEPTH || NORMAL;

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_camera_planes[6 * BLOCK_SIZE];
	__shared__ float collected_mean3d[3 * BLOCK_SIZE];
	__shared__ float collected_normals[3 * BLOCK_SIZE];
	__shared__ float collected_ts[BLOCK_SIZE];
	__shared__ float2 collected_ray_planes[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - alphas[pix_id]) : 0;
	const float w_final = inside ? alphas[pix_id] : 0;

	
	float T = T_final;
	float w = w_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	const int max_contributor = inside ? n_contrib[pix_id + H * W] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float accum_coord_rec[3] = {0};
	float dL_dpixel_coord[3];
	float accum_t_rec = 0;
	float dL_dpixel_t;
	float dL_dpixel_mt;
	float accum_alpha_rec = 0;
	float dL_dalpha;
	float accum_normal_rec[3] = {0};
	float dL_dpixel_normal[3];
	float dL_dpixel_mcoord[3];
	
	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		dL_dalpha = dL_dalphas[pix_id];

		if constexpr (GEO)
		{
			float ww = w_final*w_final;
			if constexpr (COORD)
			{
				float dL_dpixel_coord_w[3] = {dL_dpixel_coords[pix_id],
											dL_dpixel_coords[H * W + pix_id],
											dL_dpixel_coords[2 * H * W + pix_id]};
				float pixel_accum_coord[3] = {accum_coord[pix_id],
												accum_coord[H * W + pix_id],
												accum_coord[2 * H * W + pix_id]};
				for(int i = 0; i < 3; i++)
				{
					dL_dalpha -= dL_dpixel_coord_w[i]*pixel_accum_coord[i]/ww;
					dL_dpixel_coord[i] = dL_dpixel_coord_w[i] / w_final;
					dL_dpixel_mcoord[i] = dL_dpixel_mcoords[i * H * W + pix_id];
				}
			}
			if constexpr (DEPTH)
			{
				float dL_dpixel_depth_w = dL_dpixel_depths[pix_id];
				float pixel_accum_depth = accum_depth[pix_id];
				dL_dalpha -= dL_dpixel_depth_w*pixel_accum_depth/ww;
				dL_dpixel_t = dL_dpixel_depth_w / w_final/ ln;
				dL_dpixel_mt = dL_dpixel_mdepths[pix_id] / ln;
			}
			if constexpr (NORMAL)
			{
				glm::vec3 dL_dpixel_normaln = glm::vec3(dL_dpixel_normals[pix_id],
														dL_dpixel_normals[H * W + pix_id],
														dL_dpixel_normals[2 * H * W + pix_id]);
				glm::vec3 normaln = glm::vec3(normalmap[pix_id],
												normalmap[H * W + pix_id],
												normalmap[2 * H * W + pix_id]);
				float normal_len = normal_length[pix_id];
				glm::vec3 dL;
				if(normal_len<NORMALIZE_EPS)
					dL = dL_dpixel_normaln/NORMALIZE_EPS;
				else
					dL = (dL_dpixel_normaln - glm::dot(dL_dpixel_normaln,normaln)*normaln)/normal_len;
				for (int i = 0; i < 3; i++)
					dL_dpixel_normal[i] = dL[i];
			}
		}
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_coord[3] = { 0 };
	float last_t = 0;
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
			if constexpr (COORD)
			{
				for(int ch = 0; ch < 6; ch++)
					collected_camera_planes[ch * BLOCK_SIZE + block.thread_rank()] = camera_planes[coll_id * 6 + ch];
				for(int ch = 0; ch < 3; ch++)
					collected_mean3d[ch * BLOCK_SIZE + block.thread_rank()] = view_points[coll_id * 3 + ch];
			}
			if constexpr (DEPTH)
			{
				collected_ray_planes[block.thread_rank()] = ray_planes[coll_id];
				collected_ts[block.thread_rank()] = ts[coll_id];
			}
			if constexpr (NORMAL)
			{
				float3 normal = normals[coll_id];
				collected_normals[0 * BLOCK_SIZE + block.thread_rank()] = normal.x;
				collected_normals[1 * BLOCK_SIZE + block.thread_rank()] = normal.y;
				collected_normals[2 * BLOCK_SIZE + block.thread_rank()] = normal.z;
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


			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			// const float alpha = con_o.w * G;
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			const float& dpixel_t_dt = dchannel_dcolor;

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
			
			float dL_dcoords[3];
			float dL_dt;
			float2 camera_plane0;
			float2 camera_plane1;
			float2 camera_plane2;
			float2 ray_plane;
			if constexpr (COORD)
			{
				camera_plane0 = {collected_camera_planes[j], collected_camera_planes[j + BLOCK_SIZE]};
				camera_plane1 = {collected_camera_planes[j + BLOCK_SIZE * 2], collected_camera_planes[j + BLOCK_SIZE * 3]};
				camera_plane2 = {collected_camera_planes[j + BLOCK_SIZE * 4], collected_camera_planes[j + BLOCK_SIZE * 5]};
				float coord[3] = {collected_mean3d[j] + camera_plane0.x * d.x + camera_plane0.y * d.y,
									collected_mean3d[j + BLOCK_SIZE] + camera_plane1.x * d.x + camera_plane1.y * d.y,
									collected_mean3d[j + BLOCK_SIZE * 2] + camera_plane2.x * d.x + camera_plane2.y * d.y};

				for (int ch = 0; ch < 3; ch++)
				{
					const float c = coord[ch];
					// Update last color (to be used in the next iteration)
					accum_coord_rec[ch] = last_alpha * last_coord[ch] + (1.f - last_alpha) * accum_coord_rec[ch];
					last_coord[ch] = c;

					const float dL_dchannel = dL_dpixel_coord[ch];
					dL_dopa += (c - accum_coord_rec[ch]) * dL_dchannel;
					// Update the gradients w.r.t. normal of the Gaussian. 
					// Atomic, since this pixel is just one of potentially
					// many that were affected by this Gaussian.
					dL_dcoords[ch] = dchannel_dcolor * dL_dchannel;
					if (contributor == max_contributor-1) {
						dL_dcoords[ch] += dL_dpixel_mcoord[ch];
					}
				}

				atomicAdd(&(dL_dmean3D[global_id].x), dL_dcoords[0]);
				atomicAdd(&(dL_dmean3D[global_id].y), dL_dcoords[1]);
				atomicAdd(&(dL_dmean3D[global_id].z), dL_dcoords[2]);
				atomicAdd(&dL_dcamera_planes[global_id*6], dL_dcoords[0] * d.x / focal_x);
				atomicAdd(&dL_dcamera_planes[global_id*6+1], dL_dcoords[0] * d.y / focal_y);
				atomicAdd(&dL_dcamera_planes[global_id*6+2], dL_dcoords[1] * d.x / focal_x);
				atomicAdd(&dL_dcamera_planes[global_id*6+3], dL_dcoords[1] * d.y / focal_y);
				atomicAdd(&dL_dcamera_planes[global_id*6+4], dL_dcoords[2] * d.x / focal_x);
				atomicAdd(&dL_dcamera_planes[global_id*6+5], dL_dcoords[2] * d.y / focal_y);
			}

			if constexpr (DEPTH)
			{
				const float t_center = collected_ts[j];
				ray_plane = collected_ray_planes[j];
				float t = t_center + (ray_plane.x * d.x + ray_plane.y * d.y);
				accum_t_rec = last_alpha * last_t + (1.f - last_alpha) * accum_t_rec;
				last_t = t;
				dL_dopa += (t - accum_t_rec) * dL_dpixel_t;
				dL_dt = dpixel_t_dt * dL_dpixel_t;
				if (contributor == max_contributor-1) {
					dL_dt += dL_dpixel_mt;
				}
				
				atomicAdd(&(dL_dts[global_id]), dL_dt);
				atomicAdd(&dL_dray_planes[global_id].x, dL_dt * d.x / focal_x);
				atomicAdd(&dL_dray_planes[global_id].y, dL_dt * d.y / focal_y);
			}

			if constexpr (NORMAL)
			{
				for (int ch = 0; ch < 3; ch++)
				{
					const float c = collected_normals[ch * BLOCK_SIZE + j];
					// Update last color (to be used in the next iteration)
					accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
					last_normal[ch] = c;

					const float dL_dchannel = dL_dpixel_normal[ch];
					dL_dopa += (c - accum_normal_rec[ch]) * dL_dchannel;
					// Update the gradients w.r.t. normal of the Gaussian. 
					// Atomic, since this pixel is just one of potentially
					// many that were affected by this Gaussian.
					atomicAdd(&(dL_dnormals[global_id * 3 + ch]), dchannel_dcolor * dL_dchannel);
				}
			}

			// Propagate gradients from pixel alpha (weights_sum) to opacity
			accum_alpha_rec = last_alpha + (1.f - last_alpha) * accum_alpha_rec;
			dL_dopa += (1 - accum_alpha_rec) * dL_dalpha;


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
			float dL_ddelx = dL_dG * dG_ddelx;
			float dL_ddely = dL_dG * dG_ddely;
			if constexpr (COORD)
			{
				dL_ddelx += dL_dcoords[0] * camera_plane0.x
							+ dL_dcoords[1] * camera_plane1.x
							+ dL_dcoords[2] * camera_plane2.x;
				dL_ddely += dL_dcoords[0] * camera_plane0.y
							+ dL_dcoords[1] * camera_plane1.y
							+ dL_dcoords[2] * camera_plane2.y;
			}
			if constexpr (DEPTH)
			{
				dL_ddelx += dL_dt * ray_plane.x;
				dL_ddely += dL_dt * ray_plane.y;
			}
			atomicAdd(&dL_dmean2D[global_id].x, dL_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_ddely * ddely_dy);
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
	const float kernel_size,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float3* dL_dview_points,
	glm::vec3* dL_dmean3D,
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
		kernel_size,
		viewmatrix,
		dL_dconic,
		dL_dcamera_plane,
		dL_dray_plane,
		dL_dnormals,
		(glm::vec3*)dL_dmean3D,
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
		dL_dview_points,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dts,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

// the Bool inputs can be replaced by an enumeration variable for different functions.
void BACKWARD::render(
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
	bool require_depth)
{
#define RENDER_CUDA_CALL(template_coord, template_depth, template_normal) \
    renderCUDA<NUM_CHANNELS, template_coord, template_depth, template_normal> <<<grid, block>>> ( \
        ranges, point_list, W, H, bg_color, view_points, means2D, conic_opacity, colors, \
        depths, ts, camera_planes, ray_planes, alphas, normals, \
		accum_coord, accum_depth, normal_length, \
        n_contrib, dL_dpixels, dL_dpixel_coords, dL_dpixel_mcoords, dL_dpixel_depth, \
        dL_dpixel_mdepth, dL_dalphas, dL_dpixel_normals, normalmap, \
        focal_x, focal_y, dL_dmean3D, dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, \
        dL_dts, dL_dcamera_planes, dL_dray_planes, dL_dnormals)

	if (require_coord && require_depth)
		RENDER_CUDA_CALL(true, true, true);
	else if (require_coord && !require_depth)
		RENDER_CUDA_CALL(true, false, true);
	else if(!require_coord && require_depth)
		RENDER_CUDA_CALL(false, true, true);
	else
		RENDER_CUDA_CALL(false, false, false);

#undef RENDER_CUDA_CALL
}