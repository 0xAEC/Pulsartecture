// ultra_rwka/backend/cuda/idam/kernel.cuh
#ifndef ULTRA_RWKA_BACKEND_CUDA_IDAM_KERNEL_CUH_
#define ULTRA_RWKA_BACKEND_CUDA_IDAM_KERNEL_CUH_

#include <torch/extension.h> // Includes ATen headers, cuda_runtime.h, cuda_fp16.h etc.
#include <vector>
#include <limits>      // For std::numeric_limits
#include <type_traits> // For std::is_same_v

// --- Configurable Constants ---
// These might be needed by host code (e.g., to check constraints before launch)
// or by utility functions defined/used across files.
constexpr int IDAM_CUDA_BLOCK_SIZE = 256;
constexpr float IDAM_CUDA_EPSILON = 1e-9f;
constexpr int TILE_N = 16;
constexpr int TILE_DV = 16;

// --- Vectorization Types ---
// Define vector types based on scalar_t, needed by device utils/kernels
template <typename scalar_t> struct VecType {};
template <> struct VecType<float> { using Type4 = float4; using Type2 = float2; };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530 // FP16 compute support
template <> struct VecType<__half> {
    using VecLoadStore4 = float2; // Load 4 bytes (2x half)
    using VecType4 = half2;      // Operate on 2 halves
    using VecLoadStore2 = half2; // Load 2 bytes (1x half2)
    using VecType2 = half2;      // Operate on 2 halves
};
#endif

// --- Utility Functions Namespace (Device Functions) ---
namespace utils {

// Helper to cast dynamic shared memory pointers
template <typename T>
__device__ __forceinline__ T *AsDynamicSharedMemory(unsigned char *ptr) {
    return reinterpret_cast<T *>(ptr);
}

// Max Operator (required by reductions and potentially others)
template <typename scalar_t>
__device__ __forceinline__ scalar_t CudaMax(scalar_t a, scalar_t b);

template <> __device__ __forceinline__ float CudaMax<float>(float a, float b) { return fmaxf(a, b); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
template <> __device__ __forceinline__ __half CudaMax<__half>(__half a, __half b) { return __hmax(a, b); }
#endif

// --- Optimized Block Reductions ---
// Warp-level reductions (definitions required for inline expansion)
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val);

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
template <>
__device__ __forceinline__ __half warp_reduce_sum<__half>(__half val_h);
#endif

// Block-level reductions (definitions required for inline expansion)
template <typename scalar_t, int BLOCK_SIZE>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t val, scalar_t* shared_reduce_tmp);

template <typename scalar_t, int BLOCK_SIZE>
__device__ __forceinline__ scalar_t block_reduce_max(scalar_t val, scalar_t* shared_reduce_tmp);


// --- Vectorized Squared Euclidean Distance ---
// (definition required for inline expansion)
template <typename scalar_t>
__device__ __forceinline__ scalar_t vectorized_sq_distance(
    const scalar_t* __restrict__ v1,
    const scalar_t* __restrict__ v2,
    int dim);

// --- Load Shared Learnable Keys ---
// (definition required for inline expansion)
template <typename scalar_t, int BLOCK_SIZE>
__device__ void load_shared_lkeys(
    const scalar_t* __restrict__ learnable_keys_ptr,
    scalar_t* shared_lkeys,
    int NumBins, int D_key,
    int lkeys_stride_N, int lkeys_stride_D,
    bool keys_are_shared); // Consider removing keys_are_shared if truly unused

// --- Definitions for Device Inline Functions ---
// (Place definitions here so they are available when kernel.cu is compiled)

// Warp-level Reductions Definitions
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = CudaMax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
template <>
__device__ __forceinline__ __half warp_reduce_sum<__half>(__half val_h) {
    float val_f = static_cast<float>(val_h);
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val_f += __shfl_down_sync(0xFFFFFFFF, val_f, offset);
    }
    return static_cast<__half>(val_f);
}
#endif

// Block-level Reductions Definitions
template <typename scalar_t, int BLOCK_SIZE>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t val, scalar_t* shared_reduce_tmp) {
    static_assert((BLOCK_SIZE % warpSize == 0), "BLOCK_SIZE must be a multiple of warpSize for this reduction.");
    constexpr int num_warps = BLOCK_SIZE / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warp_reduce_sum(val); // Warp level reduction

    if (lane_id == 0) shared_reduce_tmp[warp_id] = val;
    __syncthreads();

    val = (lane_id < num_warps) ? shared_reduce_tmp[lane_id] : static_cast<scalar_t>(0.0);
     if constexpr (std::is_same_v<scalar_t, __half>) { // Handle default value for half
         if (lane_id >= num_warps) val = static_cast<__half>(0.0f);
     }

    if (warp_id == 0) {
        // Reduce within the first warp (special handling for __half)
        if constexpr (std::is_same_v<scalar_t, __half>) {
             float val_f = static_cast<float>(val);
             #pragma unroll
             for (int offset = warpSize / 2; offset >= 1; offset /= 2) {
                 if (lane_id < num_warps) { // Ensure active lanes from first reduction step
                     val_f += __shfl_down_sync(0xFFFFFFFF, val_f, offset, warpSize);
                 }
             }
             if (lane_id == 0) shared_reduce_tmp[0] = static_cast<__half>(val_f);
        } else {
             #pragma unroll
             for (int offset = warpSize / 2; offset >= 1; offset /= 2) {
                 if (lane_id < num_warps) {
                     val += __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize);
                 }
             }
             if (lane_id == 0) shared_reduce_tmp[0] = val;
        }
    }
    __syncthreads();
    return shared_reduce_tmp[0]; // All threads return the broadcasted result
}


template <typename scalar_t, int BLOCK_SIZE>
__device__ __forceinline__ scalar_t block_reduce_max(scalar_t val, scalar_t* shared_reduce_tmp) {
    static_assert((BLOCK_SIZE % warpSize == 0), "BLOCK_SIZE must be a multiple of warpSize for this reduction.");
    constexpr int num_warps = BLOCK_SIZE / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    val = warp_reduce_max(val); // Warp level reduction

    if (lane_id == 0) shared_reduce_tmp[warp_id] = val;
    __syncthreads();

    scalar_t default_min = -std::numeric_limits<scalar_t>::max();
     if constexpr (std::is_same_v<scalar_t, __half>) {
         default_min = static_cast<scalar_t>(-65504.0f);
     }
    val = (lane_id < num_warps) ? shared_reduce_tmp[lane_id] : default_min;

    if (warp_id == 0) {
        #pragma unroll
        for (int offset = warpSize / 2; offset >= 1; offset /= 2) {
             if (lane_id < num_warps) {
                 val = CudaMax(val, __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize));
             }
        }
        if (lane_id == 0) shared_reduce_tmp[0] = val;
    }
    __syncthreads();
    return shared_reduce_tmp[0]; // All threads return the broadcasted result
}


// Vectorized Squared Euclidean Distance Definition
template <typename scalar_t>
__device__ __forceinline__ scalar_t vectorized_sq_distance(
    const scalar_t* __restrict__ v1,
    const scalar_t* __restrict__ v2,
    int dim)
{
    float acc = 0.0f; // Use float accumulator
    int d = 0;

    if constexpr (std::is_same_v<scalar_t, float>) {
        using Vec4 = typename VecType<scalar_t>::Type4;
        const int vec_size_4 = 4;
        if (dim >= vec_size_4) {
            for (; d <= dim - vec_size_4; d += vec_size_4) {
                Vec4 v1_vec = *reinterpret_cast<const Vec4*>(v1 + d);
                Vec4 v2_vec = *reinterpret_cast<const Vec4*>(v2 + d);
                float4 diff_vec = {v1_vec.x - v2_vec.x, v1_vec.y - v2_vec.y, v1_vec.z - v2_vec.z, v1_vec.w - v2_vec.w};
                acc += diff_vec.x * diff_vec.x + diff_vec.y * diff_vec.y + diff_vec.z * diff_vec.z + diff_vec.w * diff_vec.w;
            }
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    else if constexpr (std::is_same_v<scalar_t, __half>) {
        using VecLoadStore4 = typename VecType<scalar_t>::VecLoadStore4; // float2
        using VecType4 = typename VecType<scalar_t>::VecType4;          // half2
        using VecLoadStore2 = typename VecType<scalar_t>::VecLoadStore2; // half2
        using VecType2 = typename VecType<scalar_t>::VecType2;          // half2

        const int vec_size_4 = 4;
        if (dim >= vec_size_4) {
            for (; d <= dim - vec_size_4; d += vec_size_4) {
                VecLoadStore4 v1_load = *reinterpret_cast<const VecLoadStore4*>(v1 + d);
                VecLoadStore4 v2_load = *reinterpret_cast<const VecLoadStore4*>(v2 + d);
                VecType4 v1_vec_lo = __low2half2(v1_load); VecType4 v1_vec_hi = __high2half2(v1_load);
                VecType4 v2_vec_lo = __low2half2(v2_load); VecType4 v2_vec_hi = __high2half2(v2_load);
                VecType4 diff_vec_lo = __hsub2(v1_vec_lo, v2_vec_lo); VecType4 diff_vec_hi = __hsub2(v1_vec_hi, v2_vec_hi);
                VecType4 sq_diff_lo = __hmul2(diff_vec_lo, diff_vec_lo); VecType4 sq_diff_hi = __hmul2(diff_vec_hi, diff_vec_hi);
                acc += static_cast<float>(sq_diff_lo.x) + static_cast<float>(sq_diff_lo.y) + static_cast<float>(sq_diff_hi.x) + static_cast<float>(sq_diff_hi.y);
            }
        }
        const int vec_size_2 = 2;
        if (d <= dim - vec_size_2) {
            VecLoadStore2 v1_vec = *reinterpret_cast<const VecLoadStore2*>(v1 + d);
            VecLoadStore2 v2_vec = *reinterpret_cast<const VecLoadStore2*>(v2 + d);
            VecType2 diff_vec = __hsub2(v1_vec, v2_vec);
            VecType2 sq_diff = __hmul2(diff_vec, diff_vec);
            acc += static_cast<float>(sq_diff.x) + static_cast<float>(sq_diff.y);
            d += vec_size_2;
        }
    }
#endif
    // Scalar tail loop
    for (; d < dim; ++d) {
        float diff = static_cast<float>(v1[d]) - static_cast<float>(v2[d]);
        acc += diff * diff;
    }
    return static_cast<scalar_t>(acc);
}


// Load Shared Learnable Keys Definition
template <typename scalar_t, int BLOCK_SIZE>
__device__ __forceinline__ void load_shared_lkeys( // Mark as inline
    const scalar_t* __restrict__ learnable_keys_ptr,
    scalar_t* shared_lkeys,
    int NumBins, int D_key,
    int lkeys_stride_N, int lkeys_stride_D,
    bool keys_are_shared) // Keep param for API consistency, even if unused internally now
{
    int total_elements = NumBins * D_key;
    for (int idx = threadIdx.x; idx < total_elements; idx += BLOCK_SIZE) {
        int n_bin = idx / D_key;
        int d = idx % D_key;
        int global_offset = n_bin * lkeys_stride_N + d * lkeys_stride_D;
        shared_lkeys[idx] = learnable_keys_ptr[global_offset];
    }
}

} // namespace utils


// --- Host-Callable Function Declarations ---
namespace ultra_rwka {
namespace backend {
namespace cuda {
namespace idam {

// Function to perform i-DAM retrieval step
std::tuple<torch::Tensor, torch::Tensor> idam_retrieve_cuda(
    const torch::Tensor& query_keys,      // (B, Dk) or (B, T, Dk)
    const torch::Tensor& buffer,          // (B, Nb, Dv)
    const torch::Tensor& learnable_keys,  // (Nb, Dk) or (B, Nb, Dk)
    const torch::Tensor& temperature);    // Scalar tensor

// Function to perform i-DAM update step
void idam_update_cuda(
    torch::Tensor& buffer,                // In/Out: (B, Nb, Dv)
    const torch::Tensor& attn_weights,    // In: (B, T, Nb) or (B, Nb)
    const torch::Tensor& values,          // In: (B, T, Dv) or (B, Dv)
    float lambda_buf,
    torch::Tensor& learnable_keys,      // In/Out: (Nb, Dk) or (B, Nb, Dk)
    const torch::Tensor& keys,            // In: (B, T, Dk) or (B, Dk)
    float lambda_keys);

// Note: Kernel declarations (__global__ functions) are typically NOT placed
// in the header file as they are implementation details of the .cu file.
// The host functions above are the public interface.

} // namespace idam
} // namespace cuda
} // namespace backend
} // namespace ultra_rwka


#endif // ULTRA_RWKA_BACKEND_CUDA_IDAM_KERNEL_CUH_
