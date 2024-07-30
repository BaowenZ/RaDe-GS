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

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <glm/glm.hpp>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
#define NEAR_PLANE 0.2
#define FAR_PLANE 100.0
#define NORMALIZE_EPS 1.0E-12F

#define DEPTH_OFFSET 6
#define ALPHA_OFFSET 7
#define DISTORTION_OFFSET 8
#define OUTPUT_CHANNELS 9

// #define MAX_NUM_CONTRIBUTORS 256
#define MAX_NUM_CONTRIBUTORS 512
#define MAX_NUM_PROJECTED 256

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

// __device__ const float kernel_size = 0.1;
// __device__ const float kernel_size = 0.0;

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

namespace glm_modification
{
	// Incorporate the transferSign, pythag, equal, and findEigenvaluesSymReal functions from the glm library, 
	// with small modifications on findEgienvaluesSymReal to ensure numerical stability for big Gaussian kernels.
	// https://github.com/g-truc/glm/blob/33b4a621a697a305bc3a7610d290677b96beb181/glm/gtx/pca.inl
	// https://github.com/g-truc/glm/blob/33b4a621a697a305bc3a7610d290677b96beb181/glm/ext/scalar_relational.inl
	template<typename genType>
	__forceinline__ __device__ bool equal(genType const& x, genType const& y, genType const& epsilon)
	{
		return abs(x - y) <= epsilon;
	}

	template<typename T>
	__forceinline__ __device__ static T transferSign(T const& v, T const& s)
	{
		return ((s) >= 0 ? glm::abs(v) : -glm::abs(v));
	}

	template<typename T>
	__forceinline__ __device__ static T pythag(T const& a, T const& b) {
		static const T epsilon = static_cast<T>(0.0000001);
		T absa = glm::abs(a);
		T absb = glm::abs(b);
		if(absa > absb) {
			absb /= absa;
			absb *= absb;
			return absa * glm::sqrt(static_cast<T>(1) + absb);
		}
		if(glm_modification::equal<T>(absb, 0, epsilon)) return static_cast<T>(0);
		absa /= absb;
		absa *= absa;
		return absb * glm::sqrt(static_cast<T>(1) + absa);
	}


	template<glm::length_t D, typename T, glm::qualifier Q>
	__forceinline__ __device__ unsigned int findEigenvaluesSymReal
	(
		glm::mat<D, D, T, Q> const& covarMat,
		glm::vec<D, T, Q>& outEigenvalues,
		glm::mat<D, D, T, Q>& outEigenvectors
	)
	{

		T a[D * D]; // matrix -- input and workspace for algorithm (will be changed inplace)
		T d[D]; // diagonal elements
		T e[D]; // off-diagonal elements

		for(glm::length_t r = 0; r < D; r++)
			for(glm::length_t c = 0; c < D; c++)
				a[(r) * D + (c)] = covarMat[c][r];

		// 1. Householder reduction.
		glm::length_t l, k, j, i;
		T scale, hh, h, g, f;
		static const T epsilon = static_cast<T>(0.0000001);

		for(i = D; i >= 2; i--)
		{
			l = i - 1;
			h = scale = 0;
			if(l > 1)
			{
				for(k = 1; k <= l; k++)
				{
					scale += glm::abs(a[(i - 1) * D + (k - 1)]);
				}
				if(glm_modification::equal<T>(scale, 0, epsilon))
				{
					e[i - 1] = a[(i - 1) * D + (l - 1)];
				}
				else
				{
					for(k = 1; k <= l; k++)
					{
						a[(i - 1) * D + (k - 1)] /= scale;
						h += a[(i - 1) * D + (k - 1)] * a[(i - 1) * D + (k - 1)];
					}
					f = a[(i - 1) * D + (l - 1)];
					g = ((f >= 0) ? -glm::sqrt(h) : glm::sqrt(h));
					e[i - 1] = scale * g;
					h -= f * g;
					a[(i - 1) * D + (l - 1)] = f - g;
					f = 0;
					for(j = 1; j <= l; j++)
					{
						a[(j - 1) * D + (i - 1)] = a[(i - 1) * D + (j - 1)] / h;
						g = 0;
						for(k = 1; k <= j; k++)
						{
							g += a[(j - 1) * D + (k - 1)] * a[(i - 1) * D + (k - 1)];
						}
						for(k = j + 1; k <= l; k++)
						{
							g += a[(k - 1) * D + (j - 1)] * a[(i - 1) * D + (k - 1)];
						}
						e[j - 1] = g / h;
						f += e[j - 1] * a[(i - 1) * D + (j - 1)];
					}
					hh = f / (h + h);
					for(j = 1; j <= l; j++)
					{
						f = a[(i - 1) * D + (j - 1)];
						e[j - 1] = g = e[j - 1] - hh * f;
						for(k = 1; k <= j; k++)
						{
							a[(j - 1) * D + (k - 1)] -= (f * e[k - 1] + g * a[(i - 1) * D + (k - 1)]);
						}
					}
				}
			}
			else
			{
				e[i - 1] = a[(i - 1) * D + (l - 1)];
			}
			d[i - 1] = h;
		}
		d[0] = 0;
		e[0] = 0;
		for(i = 1; i <= D; i++)
		{
			l = i - 1;
			if(!glm_modification::equal<T>(d[i - 1], 0, epsilon))
			{
				for(j = 1; j <= l; j++)
				{
					g = 0;
					for(k = 1; k <= l; k++)
					{
						g += a[(i - 1) * D + (k - 1)] * a[(k - 1) * D + (j - 1)];
					}
					for(k = 1; k <= l; k++)
					{
						a[(k - 1) * D + (j - 1)] -= g * a[(k - 1) * D + (i - 1)];
					}
				}
			}
			d[i - 1] = a[(i - 1) * D + (i - 1)];
			a[(i - 1) * D + (i - 1)] = 1;
			for(j = 1; j <= l; j++)
			{
				a[(j - 1) * D + (i - 1)] = a[(i - 1) * D + (j - 1)] = 0;
			}
		}

		// 2. Calculation of eigenvalues and eigenvectors (QL algorithm)
		glm::length_t m, iter;
		T s, r, p, dd, c, b;
		const glm::length_t MAX_ITER = 30;

		for(i = 2; i <= D; i++)
		{
			e[i - 2] = e[i - 1];
		}
		e[D - 1] = 0;

		for(l = 1; l <= D; l++)
		{
			iter = 0;
			do
			{
				for(m = l; m <= D - 1; m++)
				{
					dd = glm::abs(d[m - 1]) + glm::abs(d[m - 1 + 1]);
					if(glm_modification::equal<T>(glm::abs(e[m - 1]), 0, epsilon))
						break;
				}
				if(m != l)
				{
					if(iter++ == MAX_ITER)
					{
						return 0; // Too many iterations in FindEigenvalues
					}
					g = (d[l - 1 + 1] - d[l - 1]) / (2 * e[l - 1]);
					r = pythag<T>(g, 1);
					g = d[m - 1] - d[l - 1] + e[l - 1] / (g + transferSign(r, g));
					s = c = 1;
					p = 0;
					for(i = m - 1; i >= l; i--)
					{
						f = s * e[i - 1];
						b = c * e[i - 1];
						e[i - 1 + 1] = r = pythag(f, g);
						if(glm_modification::equal<T>(r, 0, epsilon))
						{
							d[i - 1 + 1] -= p;
							e[m - 1] = 0;
							break;
						}
						s = f / r;
						c = g / r;
						g = d[i - 1 + 1] - p;
						r = (d[i - 1] - g) * s + 2 * c * b;
						d[i - 1 + 1] = g + (p = s * r);
						g = c * r - b;
						for(k = 1; k <= D; k++)
						{
							f = a[(k - 1) * D + (i - 1 + 1)];
							a[(k - 1) * D + (i - 1 + 1)] = s * a[(k - 1) * D + (i - 1)] + c * f;
							a[(k - 1) * D + (i - 1)] = c * a[(k - 1) * D + (i - 1)] - s * f;
						}
					}
					if(glm_modification::equal<T>(r, 0, epsilon) && (i >= l))
						continue;
					d[l - 1] -= p;
					e[l - 1] = g;
					e[m - 1] = 0;
				}
			} while(m != l);
		}

		// 3. output
		for(i = 0; i < D; i++)
			outEigenvalues[i] = d[i];
		for(i = 0; i < D; i++)
			for(j = 0; j < D; j++)
				outEigenvectors[i][j] = a[(j) * D + (i)];

		return D;
	}
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif