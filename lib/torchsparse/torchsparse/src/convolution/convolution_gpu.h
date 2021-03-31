#ifndef _SPARSE_CONVOLUTION
#define _SPARSE_CONVOLUTION
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCBlas.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h> 
#include <chrono>
#include <algorithm>

void ConvolutionForwardKernelGPU(
    const float *d_in_feat, int in_nchannel, float *d_out_feat,
    int out_nchannel, const float *d_kernel,
    const int* neighbor_map,
    const int* neighbor_offset,
    const int in_npoints,
    const int out_npoints,
    const int n_neighbors,
    const bool transpose,
    cublasHandle_t cuhandle, cudaStream_t stream);

// interesting fact: Chris Choy is using double to store everything in the backward pass
// in Minkowsi Engine.
void ConvolutionBackwardKernelGPU(
    const float *d_in_feat, float *d_grad_in_feat, int in_nchannel,
    const float *d_grad_out_feat, int out_nchannel, float *d_kernel,
    float *d_grad_kernel, const int * neighbor_map,
    const int * neighbor_offset,
    const int in_npoints,
    const int out_npoints,
    const int n_neighbors,
    const bool transpose,
    cublasHandle_t cuhandle, cudaStream_t stream);

void scatter_launch(const int n_in, const int n_out, const int c, 
                               const float *in_feat, float *out_feat, 
                               const int *kmap, const bool transpose);

void gather_launch(const int n_k, const int n_in, const int c, 
                               const float *in_feat, float *out_feat, 
                               const int *kmap, const bool transpose);

void ConvolutionForwardGPU(at::Tensor in_feat, at::Tensor out_feat,
                           at::Tensor kernel, at::Tensor neighbor_map, 
                           at::Tensor neighbor_offset, const bool transpose);

void ConvolutionBackwardGPU(
    at::Tensor in_feat, at::Tensor grad_in_feat, at::Tensor grad_out_feat,
    at::Tensor kernel, at::Tensor grad_kernel, at::Tensor neighbor_map,
    at::Tensor neighbor_offset, const bool transpose);

#endif