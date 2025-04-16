// ultra_rwka/backend/cuda/fa_kla/kernel.cuh

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h> // Required for CUDA types even in the header if used (e.g., cudaStream_t)

// Include common utilities and macros (like CUDA_CHECK)
// This ensures that the host function can use CUDA_CHECK if needed,
// and the device kernel implementation (.cu file) can also include it.
#include "../utils/cuda_utils.cuh"

//----------------------------------------------------------------------------
// Namespace Definition
//----------------------------------------------------------------------------
namespace ultra_rwka {
namespace backend {
namespace cuda {
namespace fa_kla {

//----------------------------------------------------------------------------
// Host-Side Function Declaration (Callable from C++ Bindings)
//----------------------------------------------------------------------------

/**
 * @brief Host-side function to launch the FA-KLA CUDA kernel.
 *
 * This function serves as the C++ entry point called from the Python bindings.
 * It performs necessary setup (validations, grid/block calculations, stream management)
 * and launches the `fa_kla_attention_kernel` on the GPU.
 *
 * @param q_mapped Contiguous tensor containing mapped queries.
 *                 Expected layout optimized for CUDA, e.g., (B, H, T, Df).
 * @param k_mapped Contiguous tensor containing mapped keys.
 *                 Expected layout optimized for CUDA, e.g., (B, H, T, Df).
 * @param v_reshaped Contiguous tensor containing reshaped values.
 *                   Expected layout optimized for CUDA, e.g., (B, H, T, Dv).
 * @param temp Contiguous tensor containing temperatures.
 *             Expected layout optimized for CUDA, e.g., (B, H, T).
 *
 * @return torch::Tensor The output tensor containing the attention results,
 *                      with the same layout as v_reshaped, e.g., (B, H, T, Dv).
 *
 * @note Input tensors MUST be on a CUDA device and contiguous.
 * @note Assumes tensor layouts are optimized for kernel performance (e.g., B, H, T, D).
 *       Transposition might be needed before calling this function if original layout differs.
 */
torch::Tensor fa_kla_attention_cuda(
    const torch::Tensor& q_mapped,
    const torch::Tensor& k_mapped,
    const torch::Tensor& v_reshaped,
    const torch::Tensor& temp);


//----------------------------------------------------------------------------
// Device-Side Kernel Declaration (Implemented in kernel.cu)
//----------------------------------------------------------------------------

/**
 * @brief CUDA kernel performing the core FA-KLA computation.
 *
 * This __global__ function executes on the GPU. It computes the
 * temperature-modulated kernelized linear attention using techniques inspired
 * by parallel scan or FlashAttention-style tiling for efficiency.
 *
 * @tparam scalar_t The data type (e.g., float, __half).
 *
 * @param q_ptr Pointer to the query tensor data.
 * @param k_ptr Pointer to the key tensor data.
 * @param v_ptr Pointer to the value tensor data.
 * @param temp_ptr Pointer to the temperature tensor data.
 * @param out_ptr Pointer to the output tensor data (written by the kernel).
 * @param B Batch size.
 * @param T Sequence length.
 * @param H Number of heads.
 * @param Df Feature dimension per head (d_k).
 * @param Dv Value dimension per head (d_v).
 * @param q_stride_B Stride for batch dim in q_mapped.
 * @param q_stride_H Stride for head dim in q_mapped.
 * @param q_stride_T Stride for time dim in q_mapped.
 * @param k_stride_B Stride for batch dim in k_mapped.
 * @param k_stride_H Stride for head dim in k_mapped.
 * @param k_stride_T Stride for time dim in k_mapped.
 * @param v_stride_B Stride for batch dim in v_reshaped.
 * @param v_stride_H Stride for head dim in v_reshaped.
 * @param v_stride_T Stride for time dim in v_reshaped.
 * @param temp_stride_B Stride for batch dim in temp.
 * @param temp_stride_H Stride for head dim in temp.
 * @param out_stride_B Stride for batch dim in output.
 * @param out_stride_H Stride for head dim in output.
 * @param out_stride_T Stride for time dim in output.
 * @param epsilon Small value added to the denominator for numerical stability.
 *
 * @note This is only the declaration. The implementation defining the kernel
 *       logic (tiling, shared memory usage, warp operations, etc.) resides
 *       in the corresponding .cu file.
 * @note Additional parameters related to kernel launch configuration or specific
 *       algorithm variations (e.g., tile sizes) might be added.
 */
template <typename scalar_t>
__global__ void fa_kla_attention_kernel(
    const scalar_t* __restrict__ q_ptr, // Use __restrict__ where applicable
    const scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const scalar_t* __restrict__ temp_ptr,
          scalar_t* __restrict__ out_ptr,
    int B, int T, int H, int Df, int Dv,
    int q_stride_B, int q_stride_H, int q_stride_T,
    int k_stride_B, int k_stride_H, int k_stride_T,
    int v_stride_B, int v_stride_H, int v_stride_T,
    int temp_stride_B, int temp_stride_H,
    int out_stride_B, int out_stride_H, int out_stride_T,
    float epsilon // Epsilon passed to kernel
);

//----------------------------------------------------------------------------
// End of Namespace
//----------------------------------------------------------------------------
} // namespace fa_kla
} // namespace cuda
} // namespace backend
} // namespace ultra_rwka
