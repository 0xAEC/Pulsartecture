// ultra_rwka/backend/cuda/idam/kernel.cu
#include "kernel.cuh" 
#include <ATen/Dispatch.h>
#include <ATen/Functions.h> // For aten::mean
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits>
#include <type_traits> 

// Configurable constants
// Consider making BLOCK_SIZE configurable or tuned based on architecture/problem
constexpr int IDAM_CUDA_BLOCK_SIZE = 256; // Power-of-2 often good for reductions
constexpr float IDAM_CUDA_EPSILON = 1e-9f; // Epsilon for softmax stability

constexpr int TILE_N = 16;  // Number of bins processed per tile iteration
constexpr int TILE_DV = 16; // Number of value dimensions processed per tile iteration

// Define vector types based on scalar_t
template <typename scalar_t> struct VecType {};
template <> struct VecType<float> { using Type4 = float4; using Type2 = float2; };
#if defined(CUDA_ARCH) && CUDA_ARCH >= 530 // FP16 compute support
template <> struct VecType<__half> {
    // Use float2 to load/store 4 bytes (2x half). Vector ops use half2.
    using VecLoadStore4 = float2; // Load 4 bytes
    using VecType4 = half2;      // Operate on 2 halves
    using VecLoadStore2 = half2; // Load 2 bytes
    using VecType2 = half2;      // Operate on 2 halves
};
#endif

// --- Utility Functions Namespace ---
namespace utils {

// Dynamic Shared Memory Helper (unchanged)
template <typename T>
__device__ __forceinline__ T *AsDynamicSharedMemory(unsigned char *ptr) {
    return reinterpret_cast<T *>(ptr);
}

// Max Operator (unchanged)
template <typename scalar_t>
__device__ __forceinline__ scalar_t CudaMax(scalar_t a, scalar_t b);

template <> __device__ __forceinline__ float CudaMax<float>(float a, float b) { return fmaxf(a, b); }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
template <> __device__ __forceinline__ __half CudaMax<__half>(__half a, __half b) { return __hmax(a, b); }
#endif

//----------------------------------------------------------------------------
// Optimized Block Reductions (Warp Shuffle + Shared Memory)
//----------------------------------------------------------------------------

// --- Warp-level Reductions ---
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    // Requires power-of-2 warp size (standard)
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val) {
    // Requires power-of-2 warp size (standard)
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = CudaMax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Specialization for __half sum to use float accumulation for precision
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

// --- Block-level Reductions ---
template <typename scalar_t, int BLOCK_SIZE>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t val, scalar_t* shared_reduce_tmp) {
    static_assert((BLOCK_SIZE % warpSize == 0), "BLOCK_SIZE must be a multiple of warpSize for this reduction.");
    constexpr int num_warps = BLOCK_SIZE / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // 1. Warp-level reduction
    val = warp_reduce_sum(val);

    // 2. Lane 0 of each warp writes result to shared memory
    if (lane_id == 0) {
        shared_reduce_tmp[warp_id] = val;
    }
    __syncthreads(); // Ensure all warp results are written

    // 3. First warp reduces the results from shared memory
    // Read only if within the number of warps
    val = (lane_id < num_warps) ? shared_reduce_tmp[lane_id] : static_cast<scalar_t>(0.0);
    if constexpr (std::is_same_v<scalar_t, __half>) { // Use float accum for half sum
         if (lane_id >= num_warps) val = static_cast<__half>(0.0f);
    }


    if (warp_id == 0) {
        // val = warp_reduce_sum(val); // Reduce within the first warp
        // Special handling for __half sum accumulation
        if constexpr (std::is_same_v<scalar_t, __half>) {
             float val_f = static_cast<float>(val);
             #pragma unroll
             for (int offset = warpSize / 2; offset >= 1; offset /= 2) {
                 // Only reduce within the number of warps active in this step
                 if (lane_id < num_warps) {
                    val_f += __shfl_down_sync(0xFFFFFFFF, val_f, offset, warpSize); // Use warpSize width
                 }
             }
             if (lane_id == 0) shared_reduce_tmp[0] = static_cast<__half>(val_f);
        } else {
             // Standard float or other types
             #pragma unroll
             for (int offset = warpSize / 2; offset >= 1; offset /= 2) {
                 // Only reduce within the number of warps active in this step
                  if (lane_id < num_warps) { // Check bounds for shuffle source
                    val += __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize);
                  }
             }
             if (lane_id == 0) shared_reduce_tmp[0] = val; // Write final sum
        }
    }
    __syncthreads(); // Ensure final result is written

    // 4. Broadcast result from shared_reduce_tmp[0] to all threads
    return shared_reduce_tmp[0];
}


template <typename scalar_t, int BLOCK_SIZE>
__device__ __forceinline__ scalar_t block_reduce_max(scalar_t val, scalar_t* shared_reduce_tmp) {
    static_assert((BLOCK_SIZE % warpSize == 0), "BLOCK_SIZE must be a multiple of warpSize for this reduction.");
    constexpr int num_warps = BLOCK_SIZE / warpSize;
    int lane_id = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // 1. Warp-level reduction
    val = warp_reduce_max(val);

    // 2. Lane 0 of each warp writes result to shared memory
    if (lane_id == 0) {
        shared_reduce_tmp[warp_id] = val;
    }
    __syncthreads(); // Ensure all warp results are written

    // 3. First warp reduces the results from shared memory
    scalar_t default_min = -std::numeric_limits<scalar_t>::max();
     if constexpr (std::is_same_v<scalar_t, __half>) {
         default_min = static_cast<scalar_t>(-65504.0f);
     }
    val = (lane_id < num_warps) ? shared_reduce_tmp[lane_id] : default_min;

    if (warp_id == 0) {
        // val = warp_reduce_max(val); // Reduce within the first warp
        #pragma unroll
        for (int offset = warpSize / 2; offset >= 1; offset /= 2) {
             // Only reduce within the number of warps active in this step
             if (lane_id < num_warps) {
                val = CudaMax(val, __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize));
             }
        }
        if (lane_id == 0) shared_reduce_tmp[0] = val; // Write final max
    }
    __syncthreads(); // Ensure final result is written

    // 4. Broadcast result from shared_reduce_tmp[0] to all threads
    return shared_reduce_tmp[0];
}

//----------------------------------------------------------------------------
// Vectorized Squared Euclidean Distance (FP16 Accumulation in Float)
//----------------------------------------------------------------------------
template <typename scalar_t>
__device__ __forceinline__ scalar_t vectorized_sq_distance(
    const scalar_t* __restrict__ v1,
    const scalar_t* __restrict__ v2,
    int dim)
{
    // Use float accumulator for potentially better precision, especially for __half
    float acc = 0.0f;
    int d = 0;

    // Process using vector type 4 if possible (float)
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
        // No Type2 for float in this setup, go directly to scalar tail
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
    // Process using vector type 2 (half2) via float2 load/store
    else if constexpr (std::is_same_v<scalar_t, __half>) {
        using VecLoadStore4 = typename VecType<scalar_t>::VecLoadStore4; // float2
        using VecType4 = typename VecType<scalar_t>::VecType4;          // half2
        using VecLoadStore2 = typename VecType<scalar_t>::VecLoadStore2; // half2
        using VecType2 = typename VecType<scalar_t>::VecType2;          // half2

        const int vec_size_4 = 4; // Process 4 half elements at a time using float2 load
        if (dim >= vec_size_4) {
            for (; d <= dim - vec_size_4; d += vec_size_4) {
                // Load 4 half values as float2
                VecLoadStore4 v1_load = *reinterpret_cast<const VecLoadStore4*>(v1 + d);
                VecLoadStore4 v2_load = *reinterpret_cast<const VecLoadStore4*>(v2 + d);

                // Interpret as two half2 vectors
                VecType4 v1_vec_lo = __low2half2(v1_load);
                VecType4 v1_vec_hi = __high2half2(v1_load);
                VecType4 v2_vec_lo = __low2half2(v2_load);
                VecType4 v2_vec_hi = __high2half2(v2_load);

                // Compute differences
                VecType4 diff_vec_lo = __hsub2(v1_vec_lo, v2_vec_lo);
                VecType4 diff_vec_hi = __hsub2(v1_vec_hi, v2_vec_hi);

                // Square and accumulate in float
                VecType4 sq_diff_lo = __hmul2(diff_vec_lo, diff_vec_lo);
                VecType4 sq_diff_hi = __hmul2(diff_vec_hi, diff_vec_hi);

                acc += static_cast<float>(sq_diff_lo.x) + static_cast<float>(sq_diff_lo.y) +
                       static_cast<float>(sq_diff_hi.x) + static_cast<float>(sq_diff_hi.y);
            }
        }

        const int vec_size_2 = 2; // Process remaining 2 half elements using half2 load
        if (dim >= vec_size_2) { // Check if there's a pair left
           if (d <= dim - vec_size_2) { // Redundant check, but safe
                VecLoadStore2 v1_vec = *reinterpret_cast<const VecLoadStore2*>(v1 + d);
                VecLoadStore2 v2_vec = *reinterpret_cast<const VecLoadStore2*>(v2 + d);
                VecType2 diff_vec = __hsub2(v1_vec, v2_vec);
                VecType2 sq_diff = __hmul2(diff_vec, diff_vec);
                acc += static_cast<float>(sq_diff.x) + static_cast<float>(sq_diff.y);
                d += vec_size_2;
            }
        }
    }
#endif

    // Process remaining elements (tail) scalar
    for (; d < dim; ++d) {
        float diff = static_cast<float>(v1[d]) - static_cast<float>(v2[d]);
        acc += diff * diff;
    }

    return static_cast<scalar_t>(acc); // Cast accumulator back to original type
}


//----------------------------------------------------------------------------
// Load Shared Learnable Keys (Robust version using strides)
//----------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE>
__device__ void load_shared_lkeys(
    const scalar_t* __restrict__ learnable_keys_ptr, // Base pointer for the relevant key block ((Nb,Dk) or (B,Nb,Dk)[b,:,:])
    scalar_t* shared_lkeys,                          // Shared memory buffer (NumBins * D_key elements)
    int NumBins, int D_key,
    int lkeys_stride_N, int lkeys_stride_D,          // Strides for the *source* tensor block (N and D dims)
    bool keys_are_shared)                            // Unused in this version, logic handled by caller pointer adjustment
{
    // Calculate total number of elements in the key block to load
    int total_elements = NumBins * D_key;

    // Grid-stride loop: each thread loads multiple elements
    for (int idx = threadIdx.x; idx < total_elements; idx += BLOCK_SIZE) {
        // Map linear shared memory index 'idx' back to logical (n_bin, d) coordinates
        int n_bin = idx / D_key;
        int d = idx % D_key;

        // Calculate the offset within the source global memory block using strides
        // learnable_keys_ptr already points to the start of the correct block (shared or batch-specific)
        int global_offset = n_bin * lkeys_stride_N + d * lkeys_stride_D;

        // Load from global memory using calculated offset and store into contiguous shared memory
        shared_lkeys[idx] = learnable_keys_ptr[global_offset];
    }
    // Note: __syncthreads() is required *after* calling this function in the kernel
    // to ensure all threads have completed loading before the data is used.
}

} // namespace utils

// --- Main Kernels Namespace ---
namespace ultra_rwka {
namespace backend {
namespace cuda {
namespace idam {

//----------------------------------------------------------------------------
// Kernel 1: i-DAM Retrieval (Optimized with Robust Vectorization and Tiling)
//----------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE>
__global__ void idam_retrieve_kernel(
    const scalar_t* __restrict__ query_keys_ptr,
    const scalar_t* __restrict__ buffer_ptr,
    const scalar_t* __restrict__ learnable_keys_ptr, // Base pointer for keys
    scalar_t* __restrict__ retrieved_values_ptr,
    scalar_t* __restrict__ attn_weights_ptr,        // Can be nullptr if not needed
    int B, int T, int NumBins, int D_key, int D_val,
    bool query_has_T, bool keys_are_shared,
    int query_stride_B, int query_stride_T, int query_stride_D,
    int buffer_stride_B, int buffer_stride_N, int buffer_stride_Dv,
    int lkeys_stride_N, int lkeys_stride_D, // Strides for learnable keys (N, D dims ONLY)
    int rtv_stride_B, int rtv_stride_T, int rtv_stride_Dv,
    int attn_stride_B, int attn_stride_T, int attn_stride_Nb,
    float temperature_inv)
{
    // --- Grid / Block Mapping & Bound Check ---
    const int b = blockIdx.x;
    int t = query_has_T ? blockIdx.y : 0;
    if (b >= B || (query_has_T && t >= T)) return;

    const int tid = threadIdx.x;
    const int block_dim = blockDim.x; // Should be BLOCK_SIZE

    // --- Shared Memory Allocation ---
    extern __shared__ unsigned char shared_mem_raw[];
    // 1. Shared learnable keys
    scalar_t* shared_lkeys = utils::AsDynamicSharedMemory<scalar_t>(shared_mem_raw);
    size_t current_offset = NumBins * D_key * sizeof(scalar_t);

    // 2. Shared logits / intermediate attention values
    scalar_t* shared_logits_attn = reinterpret_cast<scalar_t*>(shared_mem_raw + current_offset);
    current_offset += NumBins * sizeof(scalar_t);

    // 3. Temporary storage for block reductions (one element per warp)
    constexpr int reduce_tmp_elems = BLOCK_SIZE / warpSize;
    scalar_t* shared_reduce_tmp = reinterpret_cast<scalar_t*>(shared_mem_raw + current_offset);
    current_offset += reduce_tmp_elems * sizeof(scalar_t);

    // 4. Tile for buffer values during retrieval step
    constexpr int buffer_tile_elems = TILE_N * TILE_DV;
    scalar_t* shared_buffer_tile = reinterpret_cast<scalar_t*>(shared_mem_raw + current_offset);
    // Ensure current_offset calculation matches host calculation

    // --- Load Shared Learnable Keys ---
    // Determine the correct global memory pointer for this block's keys
    const scalar_t* current_lkeys_global_ptr = learnable_keys_ptr;
    if (!keys_are_shared) {
        // If keys are per-batch, the host provides the base pointer for *all* batches.
        // We need to offset to the specific batch 'b'.
        // Host calculated lkeys_stride_B for this case.
        // NOTE: The host function now passes the already-offset pointer for the non-shared case.
        // If not, the offset logic would be: current_lkeys_global_ptr += b * lkeys_stride_B;
        // Let's assume the host passes the correctly offset pointer for simplicity/consistency.
        // The strides lkeys_stride_N, lkeys_stride_D are relative to this block.
    }

    // Use the robust loading function with strides
    utils::load_shared_lkeys<scalar_t, BLOCK_SIZE>(
        current_lkeys_global_ptr, shared_lkeys, NumBins, D_key,
        lkeys_stride_N, lkeys_stride_D, keys_are_shared); // Pass strides N, D

    __syncthreads(); // <<< SYNC: Ensure keys are loaded before use

    // --- Pointer to current query key ---
    const scalar_t* current_query_ptr = query_keys_ptr + b * query_stride_B + (query_has_T ? t * query_stride_T : 0);

    // --- Calculate Distances & Logits (using vectorized distance) ---
    // Parallelize over bins using grid-stride loop
    for (int n_bin = tid; n_bin < NumBins; n_bin += block_dim) {
        // Pointer to key in contiguous shared memory
        const scalar_t* current_lkey_ptr_shared = shared_lkeys + n_bin * D_key;

        // Use vectorized distance. Assumes query key's D_key dimension is contiguous (stride_D == 1).
        // Host ensures this by calling .contiguous() on query_keys.
        scalar_t sq_dist = utils::vectorized_sq_distance<scalar_t>(
            current_query_ptr, // Pointer to global query key
            current_lkey_ptr_shared, // Pointer to shared learnable key
            D_key
        );
        // Note: query_stride_D is not explicitly used here because vectorized_sq_distance assumes contiguous inputs.
        // If query_stride_D could be != 1, a non-vectorized loop or different handling would be needed.

        shared_logits_attn[n_bin] = -sq_dist * temperature_inv;
    }
    __syncthreads(); // <<< SYNC: Ensure all distances/logits are calculated

    // --- Calculate Softmax (using efficient block reductions) ---
    // 1. Find max logit in the block
    scalar_t thread_max_logit = -std::numeric_limits<scalar_t>::max();
    if constexpr (std::is_same_v<scalar_t, __half>) {
         thread_max_logit = static_cast<scalar_t>(-65504.0f); // Approx min for fp16 safely representable
    }
    // Each thread finds max over its assigned bins
    for (int n_bin = tid; n_bin < NumBins; n_bin += block_dim) {
        thread_max_logit = utils::CudaMax(thread_max_logit, shared_logits_attn[n_bin]);
    }
    // Reduce across the block
    scalar_t block_max_logit = utils::block_reduce_max<scalar_t, BLOCK_SIZE>(thread_max_logit, shared_reduce_tmp);
    // No syncthreads needed here, block_reduce_max includes necessary syncs internally
    // and broadcasts the result implicitly by having all threads return the same value.

    // 2. Calculate sum of exps (stabilized) and store intermediate exp values
    scalar_t thread_exp_sum = static_cast<scalar_t>(0.0);
    float thread_exp_sum_f = 0.0f; // Use float accumulator for half sum

    for (int n_bin = tid; n_bin < NumBins; n_bin += block_dim) {
        scalar_t logit = shared_logits_attn[n_bin];
        scalar_t shifted_logit = logit - block_max_logit; // Use block max for stability
        scalar_t exp_val;

        // Calculate exp. Use float for intermediate exp calculation for stability/range.
        float exp_val_f = expf(static_cast<float>(shifted_logit));
        exp_val = static_cast<scalar_t>(exp_val_f);

        // Store the intermediate (unnormalized) exp value back to shared memory for later normalization
        shared_logits_attn[n_bin] = exp_val;

        // Accumulate sum
        if constexpr (std::is_same_v<scalar_t, __half>) {
            thread_exp_sum_f += exp_val_f;
        } else {
            thread_exp_sum += exp_val; // Accumulate directly for float
        }
    }

    // Reduce the sum across the block
    scalar_t block_exp_sum;
    if constexpr (std::is_same_v<scalar_t, __half>) {
        // Pass the float sum to the reduction, which handles accumulation internally
        // Note: block_reduce_sum for half expects half input, accumulates in float, returns half.
        // We need to cast our float sum back to half first. Could optimize this.
        // Alternative: Modify block_reduce_sum to accept float accumulator directly.
        // Let's stick to the API for now: Cast float sum to half, reduce.
         block_exp_sum = utils::block_reduce_sum<scalar_t, BLOCK_SIZE>(static_cast<scalar_t>(thread_exp_sum_f), shared_reduce_tmp);
    } else {
         block_exp_sum = utils::block_reduce_sum<scalar_t, BLOCK_SIZE>(thread_exp_sum, shared_reduce_tmp);
    }
     // No syncthreads needed here due to internal syncs in block_reduce_sum.

    // 3. Normalize and write attention weights
    scalar_t inv_block_exp_sum = static_cast<scalar_t>(1.0f) / (block_exp_sum + static_cast<scalar_t>(IDAM_CUDA_EPSILON));

    // Pointer to the start of the attention weights for this (b, t) pair
    scalar_t* current_attn_ptr = nullptr;
    if (attn_weights_ptr != nullptr) {
        current_attn_ptr = attn_weights_ptr + b * attn_stride_B + (query_has_T ? t * attn_stride_T : 0);
    }

    for (int n_bin = tid; n_bin < NumBins; n_bin += block_dim) {
        scalar_t attn_w = shared_logits_attn[n_bin] * inv_block_exp_sum;
        shared_logits_attn[n_bin] = attn_w; // Store normalized attention back in shared memory for retrieval step

        // Write out attention weights if requested
        if (current_attn_ptr != nullptr) {
             current_attn_ptr[n_bin * attn_stride_Nb] = attn_w;
        }
    }
    __syncthreads(); // <<< SYNC: Ensure all threads have updated shared_logits_attn with normalized weights before retrieval

    // --- Retrieve Values (Tiled Implementation) ---
    scalar_t* current_rtv_ptr = retrieved_values_ptr + b * rtv_stride_B + (query_has_T ? t * rtv_stride_T : 0);
    const scalar_t* current_buffer_ptr = buffer_ptr + b * buffer_stride_B;

    // Loop over tiles in the D_val dimension (outer loop)
    for (int dv_tile_start = 0; dv_tile_start < D_val; dv_tile_start += TILE_DV) {
        // Thread mapping within the tile (each thread handles one D_val column)
        int dv_in_tile = tid % TILE_DV; // Column index within the tile (0 to TILE_DV-1)
        int dv_global = dv_tile_start + dv_in_tile; // Global D_val index for this thread

        // Accumulator for the weighted sum for this thread's D_val column
        // Use float accumulator for precision, especially if D_val or NumBins is large
        float weighted_sum_f = 0.0f;

        // Only threads responsible for valid D_val indices within this tile participate
        if (dv_global < D_val) {
            // Loop over tiles in the NumBins dimension (inner loop)
            for (int n_tile_start = 0; n_tile_start < NumBins; n_tile_start += TILE_N) {
                 // --- Load a tile of the buffer into shared memory ---
                 // Cooperative loading: map 2D thread index (within block) to 2D tile index
                 int load_row_warp = (tid / TILE_DV) % TILE_N; // Row index within warp's view of tile (maps to N)
                 int load_col_warp = tid % TILE_DV;           // Col index within warp's view of tile (maps to Dv)

                 int load_warp_id = tid / (TILE_N * TILE_DV); // If block larger than tile
                 int loads_per_warp = (TILE_N * TILE_DV);     // Ideally BLOCK_SIZE >= TILE_N*TILE_DV

                 // Map thread ID to a linear index within the tile for loading
                 int tid_in_tile_load = tid % (TILE_N * TILE_DV); // Assume blocksize >= tile size for now
                 int load_row = tid_in_tile_load / TILE_DV; // Row index in shared tile (0 to TILE_N-1)
                 int load_col = tid_in_tile_load % TILE_DV; // Col index in shared tile (0 to TILE_DV-1)

                 int n_global_load = n_tile_start + load_row;      // Global N bin index to load from
                 int dv_global_load = dv_tile_start + load_col;    // Global Dv index to load from

                 int shared_idx = load_row * TILE_DV + load_col;   // Linear index in shared memory tile

                 // Bounds check before loading from global memory
                 if (load_row < TILE_N && n_global_load < NumBins && dv_global_load < D_val) {
                    // Use strides for global buffer access
                    shared_buffer_tile[shared_idx] = current_buffer_ptr[n_global_load * buffer_stride_N + dv_global_load * buffer_stride_Dv];
                 } else {
                    // Pad with zero if outside bounds (simplifies computation loop)
                    shared_buffer_tile[shared_idx] = static_cast<scalar_t>(0.0f);
                 }
                 __syncthreads(); // <<< SYNC: Ensure the buffer tile is loaded before computation

                 // --- Compute weighted sum using the loaded tile ---
                 // Each thread iterates through the N dimension within the loaded tile
                 // for its assigned D_val column (dv_in_tile)
                 #pragma unroll
                 for (int n_in_tile = 0; n_in_tile < TILE_N; ++n_in_tile) {
                     int n_global = n_tile_start + n_in_tile; // Global N bin index for this iteration

                     if (n_global < NumBins) { // Check if the bin is valid (redundant if padding worked, but safe)
                         // Read buffer value from shared memory tile (column dv_in_tile)
                         scalar_t buffer_val = shared_buffer_tile[n_in_tile * TILE_DV + dv_in_tile];
                         // Read corresponding *normalized* attention weight from shared memory
                         scalar_t attn_w = shared_logits_attn[n_global];

                         // Accumulate in float
                         weighted_sum_f += static_cast<float>(attn_w) * static_cast<float>(buffer_val);
                     }
                 }
                 __syncthreads(); // <<< SYNC: Ensure all threads finish using the current tile before the next tile iteration loads over it
            } // End loop over N tiles

            // Write the final accumulated weighted sum for this dv_global index to output
            // Cast float accumulator back to scalar_t for storing
            current_rtv_ptr[dv_global * rtv_stride_Dv] = static_cast<scalar_t>(weighted_sum_f);

        } // End if (dv_global < D_val)

        // No syncthreads needed here across D_val tiles, as each thread writes independently.
        // However, if the block ends here, an implicit sync occurs.

    } // End loop over D_val tiles
}


//----------------------------------------------------------------------------
// Kernel 2: i-DAM Update (Handles shared keys correctly via flag)
//----------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE>
__global__ void idam_update_kernel(
    scalar_t* __restrict__ buffer_ptr,      // Shape (B, Nb, Dv) - To be updated
    const scalar_t* __restrict__ attn_avg_ptr, // Shape (B, Nb) - Per-batch avg attn over T
    const scalar_t* __restrict__ values_avg_ptr, // Shape (B, Dv) - Per-batch avg values over T
    scalar_t* __restrict__ learnable_keys_ptr, // Shape (Nb, Dk) or (B, Nb, Dk) - To be updated ONLY IF NOT SHARED
    const scalar_t* __restrict__ keys_avg_ptr, // Shape (B, Dk) - Per-batch avg keys over T
    int B, int NumBins, int D_val, int D_key,
    int buffer_stride_B, int buffer_stride_N, int buffer_stride_Dv,
    int attn_stride_B, int attn_stride_Nb,
    int vals_stride_B, int vals_stride_Dv,
    int lkeys_stride_B, int lkeys_stride_N, int lkeys_stride_D, // Strides for learnable keys
    int keys_stride_B, int keys_stride_D,                       // Strides for keys_avg
    float lambda_buf, float lambda_keys,
    bool update_lkeys_flag, // Controls if key update happens at all
    bool keys_are_shared)   // Controls *how* key update happens (kernel vs host)
{
    // --- Grid / Block Mapping & Bound Check ---
    const int b = blockIdx.x; // Each block handles one batch item
    if (b >= B) return;
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x; // BLOCK_SIZE

    // --- Pointers for current batch item ---
    scalar_t* buffer_b_ptr = buffer_ptr + b * buffer_stride_B;           // (Nb, Dv) view
    const scalar_t* attn_b_ptr = attn_avg_ptr + b * attn_stride_B;       // (Nb,) view
    const scalar_t* vals_b_ptr = values_avg_ptr + b * vals_stride_B;     // (Dv,) view

    // --- Buffer Update (Parallelize over NumBins, Vectorize over D_val) ---
    // Each thread handles multiple bins using a grid-stride loop
    for (int n_bin = tid; n_bin < NumBins; n_bin += block_dim) {
        // Fetch the single average attention value for this bin in this batch item
        scalar_t attn_bn = attn_b_ptr[n_bin * attn_stride_Nb];
        // Calculate gates (using float for intermediate calculation is fine)
        float write_gate_f = lambda_buf * static_cast<float>(attn_bn);
        float forget_gate_f = 1.0f - write_gate_f;

        scalar_t* current_buf_row_ptr = buffer_b_ptr + n_bin * buffer_stride_N; // Pointer to buffer[b, n_bin, :]

        // Inner loop over D_val, vectorized
        int dv = 0;
        // Vector size 4 (float)
        if constexpr (std::is_same_v<scalar_t, float>) {
            using Vec4 = typename VecType<scalar_t>::Type4;
            const int vec_size_4 = 4;
            // Check contiguity for vectorization (host ensures this via .contiguous())
            if (D_val >= vec_size_4 && buffer_stride_Dv == 1 && vals_stride_Dv == 1) {
                float forget_gate_s = forget_gate_f;
                float write_gate_s = write_gate_f;
                for (; dv <= D_val - vec_size_4; dv += vec_size_4) {
                    Vec4 current_vec = *reinterpret_cast<Vec4*>(current_buf_row_ptr + dv);
                    Vec4 write_val_vec = *reinterpret_cast<const Vec4*>(vals_b_ptr + dv); // vals_avg is (B, Dv), pointer is correct
                    current_vec.x = forget_gate_s * current_vec.x + write_gate_s * write_val_vec.x;
                    current_vec.y = forget_gate_s * current_vec.y + write_gate_s * write_val_vec.y;
                    current_vec.z = forget_gate_s * current_vec.z + write_gate_s * write_val_vec.z;
                    current_vec.w = forget_gate_s * current_vec.w + write_gate_s * write_val_vec.w;
                    *reinterpret_cast<Vec4*>(current_buf_row_ptr + dv) = current_vec;
                }
            }
        }
        // Vector size 2 (half2)
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
        else if constexpr (std::is_same_v<scalar_t, __half>) {
             using VecLoadStore2 = typename VecType<scalar_t>::VecLoadStore2; // half2
             using VecType2 = typename VecType<scalar_t>::VecType2;          // half2
             const int vec_size_2 = 2;
             if (D_val >= vec_size_2 && buffer_stride_Dv == 1 && vals_stride_Dv == 1) {
                // Convert float gates to half _once_ per bin
                __half forget_gate_h = __float2half_rn(forget_gate_f);
                __half write_gate_h = __float2half_rn(write_gate_f);
                // Broadcast scalar gates to half2 vectors
                VecType2 forget_vec = __half2half2(forget_gate_h);
                VecType2 write_gate_vec = __half2half2(write_gate_h);

                for (; dv <= D_val - vec_size_2; dv += vec_size_2) {
                    VecLoadStore2 current_vec = *reinterpret_cast<VecLoadStore2*>(current_buf_row_ptr + dv);
                    VecLoadStore2 write_val_vec = *reinterpret_cast<const VecLoadStore2*>(vals_b_ptr + dv);
                    // result = forget * current + write * new_val
                    VecType2 term1 = __hmul2(forget_vec, current_vec);
                    VecType2 term2 = __hmul2(write_gate_vec, write_val_vec);
                    *reinterpret_cast<VecLoadStore2*>(current_buf_row_ptr + dv) = __hadd2(term1, term2);
                }
            }
        }
        #endif
        // Tail loop (scalar) for remaining elements or if not contiguous
        scalar_t forget_gate_s = static_cast<scalar_t>(forget_gate_f);
        scalar_t write_gate_s = static_cast<scalar_t>(write_gate_f);
        for (; dv < D_val; ++dv) {
             int buffer_offset = dv * buffer_stride_Dv; // Use stride
             scalar_t current_buf_val = current_buf_row_ptr[buffer_offset];
             scalar_t write_val = vals_b_ptr[dv * vals_stride_Dv]; // Use stride
             current_buf_row_ptr[buffer_offset] = forget_gate_s * current_buf_val + write_gate_s * write_val;
        }
    } // End loop over n_bin for buffer update

    // --- Learnable Key Update (Conditional) ---
    // Only proceed if the flag is set AND keys are NOT shared (per-batch update)
    if (update_lkeys_flag && !keys_are_shared) {
        // Shared keys are updated on the host after kernel completion.
        // This block handles the case where learnable_keys are (B, Nb, Dk).

        // Pointer to this batch item's learnable keys block
        scalar_t* lkeys_b_ptr = learnable_keys_ptr + b * lkeys_stride_B; // (Nb, Dk) view for this batch item
        // Pointer to this batch item's average *incoming* keys
        const scalar_t* keys_b_ptr = keys_avg_ptr + b * keys_stride_B; // (Dk,) view for this batch item

        // Parallelize over NumBins, vectorize over D_key
        for (int n_bin = tid; n_bin < NumBins; n_bin += block_dim) {
             // Fetch the same average attention value used for buffer update
             scalar_t attn_bn = attn_b_ptr[n_bin * attn_stride_Nb];
             // Calculate gates for key update
             float write_gate_f = lambda_keys * static_cast<float>(attn_bn);
             float forget_gate_f = 1.0f - write_gate_f;

             // Pointer to the start of the row for this bin's key: lkeys[b, n_bin, :]
             scalar_t* current_lkey_row_ptr = lkeys_b_ptr + n_bin * lkeys_stride_N;

             // Inner loop over D_key, vectorized
             int dk = 0;
             // Vector size 4 (float)
             if constexpr (std::is_same_v<scalar_t, float>) {
                using Vec4 = typename VecType<scalar_t>::Type4;
                const int vec_size_4 = 4;
                // Check contiguity (host ensures this)
                if (D_key >= vec_size_4 && lkeys_stride_D == 1 && keys_stride_D == 1) {
                    float forget_gate_s = forget_gate_f;
                    float write_gate_s = write_gate_f;
                    for (; dk <= D_key - vec_size_4; dk += vec_size_4) {
                        Vec4 current_vec = *reinterpret_cast<Vec4*>(current_lkey_row_ptr + dk);
                        // keys_b_ptr points to the start of (Dk,) avg keys for this batch
                        Vec4 write_key_vec = *reinterpret_cast<const Vec4*>(keys_b_ptr + dk);
                        current_vec.x = forget_gate_s * current_vec.x + write_gate_s * write_key_vec.x;
                        current_vec.y = forget_gate_s * current_vec.y + write_gate_s * write_key_vec.y;
                        current_vec.z = forget_gate_s * current_vec.z + write_gate_s * write_key_vec.z;
                        current_vec.w = forget_gate_s * current_vec.w + write_gate_s * write_key_vec.w;
                        *reinterpret_cast<Vec4*>(current_lkey_row_ptr + dk) = current_vec;
                    }
                }
             }
             // Vector size 2 (half2)
             #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
             else if constexpr (std::is_same_v<scalar_t, __half>) {
                using VecLoadStore2 = typename VecType<scalar_t>::VecLoadStore2; // half2
                using VecType2 = typename VecType<scalar_t>::VecType2;          // half2
                const int vec_size_2 = 2;
                if (D_key >= vec_size_2 && lkeys_stride_D == 1 && keys_stride_D == 1) {
                    __half forget_gate_h = __float2half_rn(forget_gate_f);
                    __half write_gate_h = __float2half_rn(write_gate_f);
                    VecType2 forget_vec = __half2half2(forget_gate_h);
                    VecType2 write_gate_vec = __half2half2(write_gate_h);
                    for (; dk <= D_key - vec_size_2; dk += vec_size_2) {
                        VecLoadStore2 current_vec = *reinterpret_cast<VecLoadStore2*>(current_lkey_row_ptr + dk);
                        VecLoadStore2 write_key_vec = *reinterpret_cast<const VecLoadStore2*>(keys_b_ptr + dk);
                        VecType2 term1 = __hmul2(forget_vec, current_vec);
                        VecType2 term2 = __hmul2(write_gate_vec, write_key_vec);
                        *reinterpret_cast<VecLoadStore2*>(current_lkey_row_ptr + dk) = __hadd2(term1, term2);
                    }
                }
             }
             #endif
             // Tail loop (scalar)
             scalar_t forget_gate_s = static_cast<scalar_t>(forget_gate_f);
             scalar_t write_gate_s = static_cast<scalar_t>(write_gate_f);
             for (; dk < D_key; ++dk) {
                 int lkey_offset = dk * lkeys_stride_D; // Use stride
                 scalar_t current_lkey_val = current_lkey_row_ptr[lkey_offset];
                 scalar_t write_key_val = keys_b_ptr[dk * keys_stride_D]; // Use stride for avg keys
                 current_lkey_row_ptr[lkey_offset] = forget_gate_s * current_lkey_val + write_gate_s * write_key_val;
             }
        } // End loop over n_bin for per-batch key update
    } // End if (update_lkeys_flag && !keys_are_shared)

    // No explicit syncthreads needed at the end of the kernel.
}


//----------------------------------------------------------------------------
// Host-Side Function Implementations (Kernel Launchers)
//----------------------------------------------------------------------------

// Helper for validation (unchanged from original, good practice)
void check_tensor_properties(const torch::Tensor& t, const std::string& name, bool check_cuda = true, bool check_contiguous = false, int min_dim = -1, int max_dim = -1) {
    if (!t.defined()) {
        // Allow undefined tensors if not strictly required (like optional outputs)
        // Depending on context, might want TORCH_CHECK here instead.
        return;
    }
    if (check_cuda) TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor.");
    if (check_contiguous) TORCH_CHECK(t.is_contiguous(), name, " must be contiguous.");
    if (min_dim > 0) TORCH_CHECK(t.dim() >= min_dim, name, " must have at least ", min_dim, " dimensions, got ", t.dim());
    if (max_dim > 0) TORCH_CHECK(t.dim() <= max_dim, name, " must have at most ", max_dim, " dimensions, got ", t.dim());
}


// idam_retrieve_cuda (Launcher for Retrieval Kernel)
std::tuple<torch::Tensor, torch::Tensor> idam_retrieve_cuda(
    const torch::Tensor& query_keys,
    const torch::Tensor& buffer,
    const torch::Tensor& learnable_keys,
    const torch::Tensor& temperature)
{
    // --- Validation ---
    check_tensor_properties(query_keys, "query_keys", true, false, 2, 3); // (B, Dk) or (B, T, Dk)
    check_tensor_properties(buffer, "buffer", true, false, 3, 3);         // (B, Nb, Dv)
    check_tensor_properties(learnable_keys, "learnable_keys", true, false, 2, 3); // (Nb, Dk) or (B, Nb, Dk)
    check_tensor_properties(temperature, "temperature", true, false, 0, 0); // Scalar
    const auto dtype = query_keys.scalar_type();
    TORCH_CHECK(dtype == buffer.scalar_type(), "Dtype mismatch: query_keys vs buffer");
    TORCH_CHECK(dtype == learnable_keys.scalar_type(), "Dtype mismatch: query_keys vs learnable_keys");
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kHalf, "Only Float32 and Float16 (Half) supported for iDAM retrieve");

    // --- Contiguity ---
    // Ensure inputs are contiguous. This simplifies kernel logic and enables vectorization.
    // WARNING: .contiguous() creates a copy if the tensor isn't already contiguous.
    // This has memory overhead and performance cost if inputs are frequently non-contiguous.
    auto query_c = query_keys.contiguous();
    auto buffer_c = buffer.contiguous();
    auto lkeys_c = learnable_keys.contiguous();

    // Check *after* making contiguous that the last dimension has stride 1, essential for vectorization.
    TORCH_CHECK(query_c.stride(-1) == 1, "query_keys (contiguous) must have stride 1 in the last dimension.");
    TORCH_CHECK(buffer_c.stride(-1) == 1, "buffer (contiguous) must have stride 1 in the last dimension.");
    TORCH_CHECK(lkeys_c.stride(-1) == 1, "learnable_keys (contiguous) must have stride 1 in the last dimension.");

    // --- Dimensions ---
    const int query_ndim = query_c.dim();
    const bool query_has_T = (query_ndim == 3);
    const int B = buffer_c.size(0);
    const int T = query_has_T ? query_c.size(1) : 1; // Effective T dimension
    const int NumBins = buffer_c.size(1);
    const int D_val = buffer_c.size(2);
    const int D_key = query_c.size(-1); // Last dim of query
    const int lkeys_ndim = lkeys_c.dim();
    const bool keys_are_shared = (lkeys_ndim == 2); // Shared if (Nb, Dk)

    // --- Shape Checks ---
    TORCH_CHECK(query_c.size(0) == B, "Batch size mismatch: query_keys vs buffer");
    TORCH_CHECK(lkeys_c.size(-1) == D_key, "Key dimension mismatch: query_keys vs learnable_keys");
    if (keys_are_shared) {
        TORCH_CHECK(lkeys_c.size(0) == NumBins, "Bin dimension mismatch: shared learnable_keys vs buffer");
    } else { // Per-batch keys
        TORCH_CHECK(lkeys_c.size(0) == B, "Batch size mismatch: per-batch learnable_keys vs buffer");
        TORCH_CHECK(lkeys_c.size(1) == NumBins, "Bin dimension mismatch: per-batch learnable_keys vs buffer");
    }

    // --- Prepare Outputs ---
    c10::IntArrayRef output_shape_rtv = query_has_T ? c10::IntArrayRef({B, T, D_val}) : c10::IntArrayRef({B, D_val});
    c10::IntArrayRef output_shape_attn = query_has_T ? c10::IntArrayRef({B, T, NumBins}) : c10::IntArrayRef({B, NumBins});
    auto retrieved_values = torch::empty(output_shape_rtv, query_c.options());
    auto attn_weights = torch::empty(output_shape_attn, query_c.options());
    // Ensure outputs are contiguous for simplicity and potential downstream use
    retrieved_values = retrieved_values.contiguous();
    attn_weights = attn_weights.contiguous();

    // --- Kernel Config ---
    const int BLOCK_SIZE = IDAM_CUDA_BLOCK_SIZE;
    dim3 grid_dim(B);
    if (query_has_T) {
        grid_dim.y = T; // Launch a block per (batch, time) item
    } // Else, grid_dim is just (B)

    const dim3 block_dim(BLOCK_SIZE);

    // Calculate shared memory size
    size_t scalar_size = query_c.element_size();
    size_t lkeys_bytes = NumBins * D_key * scalar_size;
    size_t logits_attn_bytes = NumBins * scalar_size;
    size_t reduce_tmp_bytes = (BLOCK_SIZE / warpSize) * scalar_size; // Matches new reduction needs
    size_t buffer_tile_bytes = TILE_N * TILE_DV * scalar_size;
    size_t shared_mem_bytes = lkeys_bytes + logits_attn_bytes + reduce_tmp_bytes + buffer_tile_bytes;

    // Check shared memory limit
    cudaDeviceProp* deviceProp = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(shared_mem_bytes <= deviceProp->sharedMemPerBlock,
                "iDAM retrieve kernel required shared memory (", shared_mem_bytes,
                " bytes) exceeds device limit (", deviceProp->sharedMemPerBlock, " bytes). ",
                "Try reducing NumBins, D_key, TILE_N, TILE_DV, or BLOCK_SIZE.");

    at::cuda::CUDAGuard device_guard(query_c.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float temp_val = temperature.item<float>();
    TORCH_CHECK(temp_val > 0, "Temperature must be positive");
    float temp_inv = 1.0f / temp_val;

    // --- Launch Kernel ---
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "idam_retrieve_cuda_launcher", [&] {
        using scalar_t = scalar_t; // Use the dispatched type

        // Calculate strides correctly based on contiguous tensors
        // For the kernel, we only need the strides *within* the key block it accesses.
        // The base pointer passed will be adjusted for batch if keys are not shared.
        int lkeys_stride_N_arg = keys_are_shared ? lkeys_c.stride(0) : lkeys_c.stride(1);
        int lkeys_stride_D_arg = 1; // Since contiguous (stride(-1))

        // Query strides
        int query_stride_B_arg = query_c.stride(0);
        int query_stride_T_arg = query_has_T ? query_c.stride(1) : 0;
        int query_stride_D_arg = 1; // Since contiguous

        // Buffer strides
        int buffer_stride_B_arg = buffer_c.stride(0);
        int buffer_stride_N_arg = buffer_c.stride(1);
        int buffer_stride_Dv_arg = 1; // Since contiguous

        // Retrieved value strides
        int rtv_stride_B_arg = retrieved_values.stride(0);
        int rtv_stride_T_arg = query_has_T ? retrieved_values.stride(1) : 0;
        int rtv_stride_Dv_arg = 1; // Since contiguous

        // Attention weight strides
        int attn_stride_B_arg = attn_weights.stride(0);
        int attn_stride_T_arg = query_has_T ? attn_weights.stride(1) : 0;
        int attn_stride_Nb_arg = 1; // Since contiguous

        // The base pointer for learnable keys passed to the kernel.
        // If keys are per-batch, we pass the base pointer (start of all batches).
        // The kernel will offset based on blockIdx.x (batch index 'b').
        // Correction: For simplicity, let's pass the already offset pointer if not shared.
        // const scalar_t* lkeys_base_ptr = lkeys_c.data_ptr<scalar_t>();

        idam_retrieve_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
            query_c.data_ptr<scalar_t>(),
            buffer_c.data_ptr<scalar_t>(),
            lkeys_c.data_ptr<scalar_t>(), // Pass the base pointer (kernel handles offset if needed, or adjusted here)
            retrieved_values.data_ptr<scalar_t>(),
            attn_weights.data_ptr<scalar_t>(), // Pass base pointer, kernel offsets
            B, T, NumBins, D_key, D_val,
            query_has_T,
            keys_are_shared,
            query_stride_B_arg, query_stride_T_arg, query_stride_D_arg,
            buffer_stride_B_arg, buffer_stride_N_arg, buffer_stride_Dv_arg,
            // Pass N and D strides relative to a key block
            lkeys_stride_N_arg, lkeys_stride_D_arg,
            rtv_stride_B_arg, rtv_stride_T_arg, rtv_stride_Dv_arg,
            attn_stride_B_arg, attn_stride_T_arg, attn_stride_Nb_arg,
            temp_inv
        );
    });
    AT_CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    return std::make_tuple(retrieved_values, attn_weights);
}


// idam_update_cuda (Launcher for Update Kernel + Host-side Shared Key Update)
void idam_update_cuda(
    torch::Tensor& buffer,           // Input/Output: (B, Nb, Dv) - Modified in-place
    const torch::Tensor& attn_weights, // Input: (B, T, Nb) or (B, Nb)
    const torch::Tensor& values,       // Input: (B, T, Dv) or (B, Dv)
    float lambda_buf,
    torch::Tensor& learnable_keys, // Input/Output: (Nb, Dk) or (B, Nb, Dk) - Modified conditionally
    const torch::Tensor& keys,         // Input: (B, T, Dk) or (B, Dk)
    float lambda_keys)
{
    // --- Basic Validation ---
    TORCH_CHECK(buffer.defined(), "buffer tensor must be defined for update.");
    // Require buffer contiguous for safe in-place update and kernel assumptions
    check_tensor_properties(buffer, "buffer", true, true, 3, 3);
    TORCH_CHECK(attn_weights.defined(), "attn_weights tensor must be defined for update.");
    TORCH_CHECK(values.defined(), "values tensor must be defined for update.");
    check_tensor_properties(attn_weights, "attn_weights", true, false, 2, 3);
    check_tensor_properties(values, "values", true, false, 2, 3);
    const auto dtype = buffer.scalar_type();
    TORCH_CHECK(dtype == attn_weights.scalar_type(), "Dtype mismatch: buffer vs attn_weights");
    TORCH_CHECK(dtype == values.scalar_type(), "Dtype mismatch: buffer vs values");
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kHalf, "Only Float32 and Float16 (Half) supported for iDAM update");

    // --- Key Update Validation (only if lambda_keys > 0) ---
    bool update_lkeys_flag = (lambda_keys > 0.0f && lambda_keys <= 1.0f); // Ensure lambda is valid range if used
    bool keys_are_shared = false;
    int D_key = 0; // Initialize

    if (update_lkeys_flag) {
        TORCH_CHECK(learnable_keys.defined(), "learnable_keys tensor must be defined if lambda_keys > 0");
        TORCH_CHECK(keys.defined(), "keys tensor must be defined if lambda_keys > 0");
        // Require learnable_keys contiguous for safe update (kernel or host) and kernel assumptions
        check_tensor_properties(learnable_keys, "learnable_keys", true, true, 2, 3);
        check_tensor_properties(keys, "keys", true, false, 2, 3);
        TORCH_CHECK(dtype == learnable_keys.scalar_type(), "Dtype mismatch: buffer vs learnable_keys");
        TORCH_CHECK(dtype == keys.scalar_type(), "Dtype mismatch: buffer vs keys");

        keys_are_shared = (learnable_keys.dim() == 2);
        D_key = learnable_keys.size(-1); // Get D_key from learnable_keys

        // Check last dimension contiguity for vectorization
        TORCH_CHECK(learnable_keys.stride(-1) == 1, "learnable_keys (contiguous) must have stride 1 in the last dimension.");
        // We will make 'keys' contiguous later.
    }

    // --- Contiguity for Value/Key Inputs ---
    // Ensure value/key inputs are contiguous, especially last dim for vectorization.
    // Reminder: Creates copies if inputs are not already contiguous.
    auto values_c = values.contiguous();
    TORCH_CHECK(values_c.stride(-1) == 1, "values (contiguous) must have stride 1 in the last dimension.");
    torch::Tensor keys_c; // Define outside the if block scope
    if (update_lkeys_flag) {
        keys_c = keys.contiguous();
        TORCH_CHECK(keys_c.stride(-1) == 1, "keys (contiguous) must have stride 1 in the last dimension.");
        TORCH_CHECK(keys_c.size(-1) == D_key, "Key dimension mismatch: keys vs learnable_keys");
    }

    // --- Dimensions & Shape Checks ---
    const int B = buffer.size(0);
    const int NumBins = buffer.size(1);
    const int D_val = buffer.size(2);

    TORCH_CHECK(attn_weights.size(0) == B, "Batch size mismatch: attn_weights vs buffer");
    TORCH_CHECK(values_c.size(0) == B, "Batch size mismatch: values vs buffer");
    TORCH_CHECK(attn_weights.size(-1) == NumBins, "Bin dimension mismatch: attn_weights vs buffer");
    TORCH_CHECK(values_c.size(-1) == D_val, "Value dimension mismatch: values vs buffer");

    bool inputs_have_T_dim = (attn_weights.dim() == 3);
    if (inputs_have_T_dim) {
        TORCH_CHECK(values_c.dim() == 3, "Dimension mismatch: attn_weights has T dim but values does not.");
        TORCH_CHECK(attn_weights.size(1) == values_c.size(1), "Time dimension mismatch: attn_weights vs values");
        if (update_lkeys_flag) {
             TORCH_CHECK(keys_c.defined() && keys_c.dim() == 3, "Dimension mismatch: attn_weights has T dim but keys does not.");
             TORCH_CHECK(attn_weights.size(1) == keys_c.size(1), "Time dimension mismatch: attn_weights vs keys");
        }
    } else { // Inputs are 2D (B, Nb) and (B, Dv) [and potentially (B, Dk)]
        TORCH_CHECK(values_c.dim() == 2, "Dimension mismatch: attn_weights is 2D but values is not.");
         if (update_lkeys_flag) {
             TORCH_CHECK(keys_c.defined() && keys_c.dim() == 2, "Dimension mismatch: attn_weights is 2D but keys is not.");
         }
    }

    // Perform remaining shape checks related to keys inside the update_lkeys_flag block
    if (update_lkeys_flag) {
        TORCH_CHECK(keys_c.size(0) == B, "Batch size mismatch: keys vs buffer");
        // D_key mismatch already checked during contiguity step
        if (keys_are_shared) {
            TORCH_CHECK(learnable_keys.size(0) == NumBins, "Bin dimension mismatch: shared learnable_keys vs buffer");
        } else { // Per-batch keys
            TORCH_CHECK(learnable_keys.size(0) == B, "Batch size mismatch: per-batch learnable_keys vs buffer");
            TORCH_CHECK(learnable_keys.size(1) == NumBins, "Bin dimension mismatch: per-batch learnable_keys vs buffer");
        }
    }

    // --- Calculate Inputs: Average over Time Dimension if Necessary ---
    // Use PyTorch GPU operations for efficiency. Ensure results are contiguous.
    torch::Tensor attn_avg;
    if (inputs_have_T_dim) {
        attn_avg = at::mean(attn_weights, /*dim=*/1, /*keepdim=*/false).contiguous(); // Shape (B, Nb)
    } else {
        attn_avg = attn_weights.contiguous(); // Ensure contiguous even if no mean needed
    }
    TORCH_CHECK(attn_avg.stride(-1) == 1, "Averaged attention (contiguous) must have stride 1 in last dim.");


    torch::Tensor vals_avg;
    if (inputs_have_T_dim) {
        vals_avg = at::mean(values_c, /*dim=*/1, /*keepdim=*/false).contiguous(); // Shape (B, Dv)
    } else {
        vals_avg = values_c.contiguous(); // Ensure contiguous
    }
    TORCH_CHECK(vals_avg.stride(-1) == 1, "Averaged values (contiguous) must have stride 1 in last dim.");


    torch::Tensor keys_avg; // Define outside scope
    if (update_lkeys_flag) {
        if (inputs_have_T_dim) {
            keys_avg = at::mean(keys_c, /*dim=*/1, /*keepdim=*/false).contiguous(); // Shape (B, Dk)
        } else {
            keys_avg = keys_c.contiguous(); // Ensure contiguous
        }
         TORCH_CHECK(keys_avg.stride(-1) == 1, "Averaged keys (contiguous) must have stride 1 in last dim.");
    }

    // --- Kernel Configuration ---
    const int BLOCK_SIZE = IDAM_CUDA_BLOCK_SIZE;
    const dim3 grid_dim(B); // One block per batch item
    const dim3 block_dim(BLOCK_SIZE);
    const size_t shared_mem_bytes = 0; // Update kernel doesn't use dynamic shared memory

    at::cuda::CUDAGuard device_guard(buffer.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- Launch Update Kernel ---
    // This kernel updates the buffer always.
    // It updates learnable_keys IN PLACE only if update_lkeys_flag=true AND keys_are_shared=false.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "idam_update_cuda_launcher", [&] {
        using scalar_t = scalar_t;

        // Prepare strides for contiguous tensors passed to kernel
        int buf_stride_B = buffer.stride(0);
        int buf_stride_N = buffer.stride(1);
        int buf_stride_Dv = 1; // buffer is contiguous

        int attn_avg_stride_B = attn_avg.stride(0);
        int attn_avg_stride_Nb = 1; // attn_avg is contiguous

        int vals_avg_stride_B = vals_avg.stride(0);
        int vals_avg_stride_Dv = 1; // vals_avg is contiguous

        // Strides for learnable_keys (handle shared vs per-batch)
        int lkeys_stride_B_arg = 0; // Passed to kernel, relevant only if !keys_are_shared
        int lkeys_stride_N_arg = 0;
        int lkeys_stride_D_arg = 0;
        if (update_lkeys_flag) {
            lkeys_stride_B_arg = keys_are_shared ? 0 : learnable_keys.stride(0); // Stride between batches if per-batch
            lkeys_stride_N_arg = keys_are_shared ? learnable_keys.stride(0) : learnable_keys.stride(1); // Stride between bins
            lkeys_stride_D_arg = 1; // learnable_keys is contiguous
        }

        // Strides for keys_avg (always per-batch structure (B, Dk))
        int keys_avg_stride_B_arg = 0;
        int keys_avg_stride_D_arg = 0;
        if (update_lkeys_flag) {
            keys_avg_stride_B_arg = keys_avg.stride(0);
            keys_avg_stride_D_arg = 1; // keys_avg is contiguous
        }

        idam_update_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
            buffer.data_ptr<scalar_t>(),         // Modifiable buffer
            attn_avg.data_ptr<scalar_t>(),       // Read-only avg attn (B, Nb)
            vals_avg.data_ptr<scalar_t>(),       // Read-only avg vals (B, Dv)
            update_lkeys_flag ? learnable_keys.data_ptr<scalar_t>() : nullptr, // Modifiable ONLY IF !shared
            update_lkeys_flag ? keys_avg.data_ptr<scalar_t>() : nullptr,   // Read-only avg keys (B, Dk)
            B, NumBins, D_val, D_key,
            buf_stride_B, buf_stride_N, buf_stride_Dv,
            attn_avg_stride_B, attn_avg_stride_Nb,
            vals_avg_stride_B, vals_avg_stride_Dv,
            lkeys_stride_B_arg, lkeys_stride_N_arg, lkeys_stride_D_arg, // Pass calculated strides
            keys_avg_stride_B_arg, keys_avg_stride_D_arg,               // Pass calculated strides
            lambda_buf, lambda_keys,
            update_lkeys_flag,
            keys_are_shared // Kernel uses this flag internally
        );
    });
    AT_CUDA_CHECK(cudaGetLastError()); // Check kernel launch

    // --- Host-Side Shared Key Update ---
    // This section executes ONLY if keys need updating AND they are shared.
    // The kernel launch above is asynchronous; these PyTorch ops queue AFTER it on the SAME stream.
    if (update_lkeys_flag && keys_are_shared) {
        // 1. Aggregate attn_avg and keys_avg across the batch dimension B.
        //    attn_avg: (B, Nb) -> avg_attn_over_batch: (Nb)
        //    keys_avg: (B, Dk) -> avg_keys_over_batch: (Dk)
        // Use keepdim=false as we want reduction. Ensure results are contiguous if needed.
        auto avg_attn_over_batch = at::mean(attn_avg, /*dim=*/0, /*keepdim=*/false).contiguous();
        auto avg_keys_over_batch = at::mean(keys_avg, /*dim=*/0, /*keepdim=*/false).contiguous();

        // 2. Calculate update terms using the aggregated values and broadcasting.
        //    learnable_keys shape: (Nb, Dk)
        //    avg_attn_over_batch shape: (Nb)
        //    avg_keys_over_batch shape: (Dk)

        // Unsqueeze attn to (Nb, 1) for broadcasting with (Nb, Dk) learnable_keys
        auto avg_attn_bc = avg_attn_over_batch.unsqueeze(-1); // Shape: (Nb, 1)

        // Calculate forget gate factor (element-wise), shape (Nb, 1)
        // Ensure calculations happen in float for stability, then cast if needed by tensor ops
        auto forget_gate = (1.0f - lambda_keys * avg_attn_bc).to(dtype);

        // Calculate write gate factor (element-wise), shape (Nb, 1)
        auto write_gate = (lambda_keys * avg_attn_bc).to(dtype);

        // 3. Apply the single update formula IN-PLACE to the shared learnable_keys tensor.
        // Formula: learnable_keys = forget_gate * learnable_keys + write_gate * avg_keys_over_batch
        // Broadcasting:
        //   forget_gate (Nb, 1) * learnable_keys (Nb, Dk) -> (Nb, Dk)
        //   write_gate (Nb, 1) * avg_keys_over_batch (Dk)   -> (Nb, Dk)

        // Perform update efficiently using in-place PyTorch operations:
        learnable_keys *= forget_gate; // Apply forget gate in-place
        // Calculate write term: write_gate * avg_keys_over_batch (broadcasts)
        auto write_update_term = write_gate * avg_keys_over_batch;
        learnable_keys.add_(write_update_term); // Add write update term in-place


    }
}


} // namespace idam
} // namespace cuda
} // namespace backend
} // namespace ultra_rwka

