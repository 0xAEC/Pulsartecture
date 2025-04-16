// ultra_rwka/backend/cuda/wavelets/kernel.cuh

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Include common utilities and macros
#include "../utils/cuda_utils.cuh"

//----------------------------------------------------------------------------
// Namespace Definition
//----------------------------------------------------------------------------
namespace ultra_rwka {
namespace backend {
namespace cuda {
namespace wavelets {

//----------------------------------------------------------------------------
// Host-Side Function Declaration (Callable from C++ Bindings)
//----------------------------------------------------------------------------

/**
 * @brief Host-side function to launch the Wavelet Projection CUDA kernel.
 *
 * This function is called from wavelet_bindings.cpp. It takes the input signal
 * and the dynamically generated, normalized FIR filters, launches the CUDA kernel
 * to perform the projection, and returns the resulting coefficients.
 *
 * @param input_signal Contiguous tensor representing the input signal, reshaped.
 *                     Expected Shape: (N, D_in), where N = Batch * SeqLen.
 * @param filters Contiguous tensor containing the normalized filter banks.
 *                Expected Shape: (N, NumFilters, FilterLength).
 * @param output_dim The expected dimension of the output coefficient vector per input sample.
 *                   The kernel implementation must produce exactly this many coefficients.
 *
 * @return torch::Tensor The output wavelet coefficient tensor.
 *                      Shape: (N, output_dim).
 *
 * @note Input tensors MUST be on a CUDA device and contiguous.
 * @note The specific projection algorithm (e.g., convolution, dot product, DWT level)
 *       that produces 'output_dim' coefficients is implemented within the kernel.
 */
torch::Tensor wavelet_projection_cuda(
    const torch::Tensor& input_signal, // Shape (N, D_in)
    const torch::Tensor& filters,      // Shape (N, NumFilters, FilterLength)
    int output_dim                     // Expected output dimension per N
);


//----------------------------------------------------------------------------
// Device-Side Kernel Declaration (Implemented in kernel.cu)
//----------------------------------------------------------------------------

/**
 * @brief CUDA kernel performing the core wavelet projection.
 *
 * This __global__ function executes on the GPU. Each block/thread processes
 * one or more input samples (rows in the N dimension). It applies the
 * corresponding filter bank to the input sample to generate the output coefficients.
 *
 * @tparam scalar_t The data type (e.g., float, __half).
 *
 * @param input_ptr Pointer to the input signal data (N, D_in).
 * @param filters_ptr Pointer to the filter data (N, NumFilters, FilterLength).
 * @param output_ptr Pointer to the output coefficient data (N, output_dim).
 * @param N Total number of samples (Batch * SeqLen).
 * @param D_in Input signal dimension.
 * @param NumFilters Number of filters per sample.
 * @param FilterLength Length of each filter.
 * @param OutputDim Expected output dimension per sample.
 * @param input_stride_N Stride for the N dimension in input_signal.
 * @param filters_stride_N Stride for the N dimension in filters.
 * @param filters_stride_NumF Stride for the NumFilters dimension in filters.
 * @param output_stride_N Stride for the N dimension in output.
 *
 * @note The implementation in the .cu file defines how the projection is done
 *       (e.g., using shared memory for filters/input, performing convolutions).
 * @note The kernel MUST produce exactly OutputDim coefficients per input sample N.
 */
template <typename scalar_t>
__global__ void wavelet_projection_kernel(
    const scalar_t* __restrict__ input_ptr,
    const scalar_t* __restrict__ filters_ptr,
          scalar_t* __restrict__ output_ptr,
    int N,
    int D_in,
    int NumFilters,
    int FilterLength,
    int OutputDim,
    int input_stride_N, // Often D_in if contiguous
    int filters_stride_N, // Often NumFilters * FilterLength if contiguous
    int filters_stride_NumF, // Often FilterLength if contiguous
    int output_stride_N // Often OutputDim if contiguous
    // Add other parameters like shared memory pointers if needed
);


//----------------------------------------------------------------------------
// End of Namespace
//----------------------------------------------------------------------------
} // namespace wavelets
} // namespace cuda
} // namespace backend
} // namespace ultra_rwka
