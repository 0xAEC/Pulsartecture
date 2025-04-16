// ultra_rwka/backend/cuda/utils/cuda_utils.cuh

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>      // For __half support
#include <device_launch_parameters.h> // For threadIdx, blockIdx, etc.
#include <stdint.h>         // For standard integer types
#include <limits>         // For numeric_limits
#include <cmath>          // For standard math functions (host fallback)

// Include common definitions (like error checking macros)
// Ensure common.h correctly includes CUDA headers when __CUDACC__ is defined
#include "../../include/common.h"

// Optional: Include Cooperative Groups header for advanced warp/block communication
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // Requires Volta or newer
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

//----------------------------------------------------------------------------
// Namespace Definition
//----------------------------------------------------------------------------
namespace ultra_rwka {
namespace backend {
namespace cuda {
namespace utils {

//----------------------------------------------------------------------------
// Constants (Device-Side)
//----------------------------------------------------------------------------

// Example: Mathematical constants accessible from device code
#define CUDA_UTILS_PI_F 3.14159265358979323846f
#define CUDA_UTILS_LARGE_F std::numeric_limits<float>::max()
#define CUDA_UTILS_SMALL_F std::numeric_limits<float>::lowest() // Most negative

//----------------------------------------------------------------------------
// Type Aliases (Optional but can improve readability)
//----------------------------------------------------------------------------

// Example: using float2 = ::float2; // Use built-in vector types directly
// Example: using half2 = ::half2;

//----------------------------------------------------------------------------
// Basic Math & Activation Functions (Device-Side)
//----------------------------------------------------------------------------

// Use __forceinline__ to suggest inlining for these small functions
// Use templates for type flexibility (float, double, half)

template <typename T>
__forceinline__ __device__ T CudaMax(T a, T b) {
    return a > b ? a : b;
}

template <typename T>
__forceinline__ __device__ T CudaMin(T a, T b) {
    return a < b ? a : b;
}

// Example: ReLU activation
template<typename T>
__forceinline__ __device__ T CudaRelu(T val) {
    return CudaMax(static_cast<T>(0.0), val);
}

// Example: Sigmoid activation (consider fast approximation if needed)
__forceinline__ __device__ float CudaSigmoid(float val) {
    // Use __expf for potentially faster single-precision exponentiation
    return 1.0f / (1.0f + __expf(-val));
}
__forceinline__ __device__ double CudaSigmoid(double val) {
    return 1.0 / (1.0 + ::exp(-val));
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)) // Check for half support
__forceinline__ __device__ __half CudaSigmoid(__half val) {
    // Use float intermediate for calculation, then convert back
    float val_f = __half2float(val);
    val_f = 1.0f / (1.0f + __expf(-val_f));
    return __float2half(val_f);
}
#endif

// Example: Softplus activation log(1 + exp(x))
__forceinline__ __device__ float CudaSoftplus(float val) {
    // logf(1.f + expf(val)) -> use log1pf(expf(val)) for better precision?
    // Or handle large/small values: max(x, 0) + log1p(exp(-abs(x)))
    // Simpler version for now:
    return __logf(1.0f + __expf(val));
}
__forceinline__ __device__ double CudaSoftplus(double val) {
    return ::log(1.0 + ::exp(val));
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
__forceinline__ __device__ __half CudaSoftplus(__half val) {
    float val_f = __half2float(val);
    val_f = __logf(1.0f + __expf(val_f));
    return __float2half(val_f);
}
#endif


//----------------------------------------------------------------------------
// Warp-Level Primitives (Wrappers or Helpers - often used directly)
//----------------------------------------------------------------------------

// --- Warp Shuffle Sync Functions (Requires __CUDA_ARCH__ >= 300) ---
// Often used directly: __shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync

// Example: Warp-level sum reduction
template <typename T>
__forceinline__ __device__ T WarpReduceSum(T val) {
    // Requires Kepler (SM30) or newer
    // Mask includes all threads in the warp (0xffffffff)
    constexpr unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val; // Broadcast result is in the 0th lane
}

// Example: Warp-level max reduction
template <typename T>
__forceinline__ __device__ T WarpReduceMax(T val) {
    constexpr unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = CudaMax(val, __shfl_down_sync(mask, val, offset));
    }
    return val; // Broadcast result is in the 0th lane
}


// --- Cooperative Groups (Requires __CUDA_ARCH__ >= 700, Volta+) ---
// Provides more flexible and explicit grouping (tile, coalesced_group)
// Usage Example (within a kernel):
// cg::thread_block tb = cg::this_thread_block();
// cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);
// T warp_sum = cg::reduce(warp, val, cg::plus<T>());
// T block_sum = cg::reduce(tb, val, cg::plus<T>());


//----------------------------------------------------------------------------
// Shared Memory Utilities (Example: Simple Allocation)
//----------------------------------------------------------------------------

// Helper to get a correctly typed pointer to dynamically allocated shared memory
// Usage: extern __shared__ unsigned char shared_mem_raw[];
//        float* shared_floats = AsDynamicSharedMemory<float>(shared_mem_raw);
template<typename T>
__device__ T* AsDynamicSharedMemory(void* ptr) {
    return reinterpret_cast<T*>(ptr);
}

//----------------------------------------------------------------------------
// Indexing and Grid/Block Calculation Helpers (Can be useful)
//----------------------------------------------------------------------------

// Example: Simple 1D grid stride loop helper structure
// Usage: for(int i : GridStrideLoop(N)) { /* kernel body for element i */ }
// struct GridStrideLoop {
//     const int size;
//     int current;
//     const int stride;

//     __device__ GridStrideLoop(int N) :
//         size(N),
//         current(blockIdx.x * blockDim.x + threadIdx.x),
//         stride(gridDim.x * blockDim.x) {}

//     // Iterator methods
//     __device__ int operator*() const { return current; }
//     __device__ GridStrideLoop& operator++() { current += stride; return *this; }
//     __device__ bool operator!=(const GridStrideLoop&) const { // Dummy comparison needed for range-based for
//         return current < size;
//     }
//     // Standard iterator methods (begin/end)
//     __device__ GridStrideLoop begin() const { return *this; }
//     __device__ GridStrideLoop end() const { return GridStrideLoop(size); } // End condition check relies on !=
// };
// NOTE: Range-based for loop support on device can be tricky depending on CUDA version/compiler.
//       A standard explicit loop is often safer:
//       for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) { ... }


// Function to calculate grid size needed for N elements and BlockSize threads
inline dim3 CalcGridSize(int N, int BlockSize) {
    return dim3((N + BlockSize - 1) / BlockSize);
}


//----------------------------------------------------------------------------
// End of Namespace
//----------------------------------------------------------------------------
} // namespace utils
} // namespace cuda
} // namespace backend
} // namespace ultra_rwka
