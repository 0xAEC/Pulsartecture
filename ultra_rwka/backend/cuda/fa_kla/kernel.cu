// ultra_rwka/backend/cuda/fa_kla/kernel.cu
#include "kernel.cuh" // Assuming this contains necessary includes like <torch/extension.h> if needed

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h> // For AT_CUDA_CHECK
#include <ATen/Tensor.h>          // For TORCH_CHECK, torch::Tensor
#include <ATen/ATen.h>            // For torch::empty, torch::elementSize
#include <c10/cuda/CUDAStream.h>  // For at::cuda::getCurrentCUDAStream

// Include CUB headers
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>   // For block-level scan if needed directly

#include <cuda_fp16.h> // For half precision types if used

// Vector type for memory access (assuming float for now)
using scalar4 = float4;
constexpr int VEC_SIZE = 4;

// Alignment and Block/Tile Size Configuration
constexpr int SHARED_MEM_ALIGNMENT = 16; // Align padding to 16 floats (64 bytes)
constexpr int FA_KLA_CUDA_BLOCK_SIZE = 256;
constexpr int FA_KLA_T_TILE_SIZE = 256; // Keep same as block size for current parallel scan

// Ensure T_TILE_SIZE is power of 2 for the custom scan - add check if needed
// static_assert((FA_KLA_T_TILE_SIZE & (FA_KLA_T_TILE_SIZE - 1)) == 0, "T_TILE_SIZE must be a power of 2");

namespace ultra_rwka {
namespace backend {
namespace cuda {
namespace fa_kla {

// Helper to calculate padded dimension
__host__ __device__ inline int get_padded_dim(int dim, int align) {
    // Ensure alignment is at least 1 to avoid division by zero
    align = max(align, 1);
    return (dim + align - 1) / align * align;
}

//----------------------------------------------------------------------------
// Optimized Device Helpers (Parallel Scan & Reduction)
//----------------------------------------------------------------------------

// --- Efficient Block-Wide Sum Reduction ---
template <typename scalar_t, int BLOCK_SIZE>
__device__ __forceinline__ scalar_t block_reduce_sum(scalar_t val, scalar_t* shared_reduction_buffer) {
    // Assumes BLOCK_SIZE is power of 2 and >= warpSize
    int tid = threadIdx.x;
    int lane_id = tid % warpSize;
    int warp_id = tid / warpSize;
    constexpr int num_warps = BLOCK_SIZE / warpSize;

    // Intra-warp reduction using shuffles
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Warp leaders write to shared memory
    if (lane_id == 0) {
        shared_reduction_buffer[warp_id] = val;
    }
    __syncthreads(); // Ensure all warp results are written

    // Read back into first warp's threads
    // Ensure tid < num_warps check is robust even if num_warps < warpSize
    val = (tid < num_warps) ? shared_reduction_buffer[tid] : static_cast<scalar_t>(0.0);

    // Final reduction within the first warp (if num_warps > 1)
    if (warp_id == 0 && num_warps > 1) { // Check ensures warp_id is valid
        #pragma unroll
        // Use volatile to prevent compiler optimizing away shared memory writes if intermediate results aren't reused by the same thread
        volatile scalar_t* smem_volatile = shared_reduction_buffer;
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            if (lane_id < offset) {
                 val += smem_volatile[lane_id + offset];
                 smem_volatile[lane_id] = val; // Store intermediate result
            }
             // Synchronization within the warp is implicitly handled by shfl_sync below for broadcasting
        }
    }
     // No syncthreads needed here because __shfl_sync will sync the warp.

    // Broadcast result from thread 0 of the first warp
    if (num_warps > 0) // Ensure there's at least one warp
        val = __shfl_sync(0xFFFFFFFF, val, 0, warpSize); // Broadcast within the first warp first if needed, then across warps (or just read from lane 0)
    // A simpler broadcast might be just having thread 0 write to a designated shared memory location and all others read after a syncthreads,
    // but shuffle is often faster if the value is needed in registers.

    // If other warps need the result, thread 0 could write to shared memory location 0,
    // __syncthreads(), then all threads read.
    // Or, just return the value held by thread 0 (caller must know only thread 0 is valid or sync after call).
    // Let's use the shuffle broadcast, assuming result needed in registers by all threads.
    // Need to ensure all threads participate in the final broadcast if needed.

    // Re-evaluate broadcast: The shuffle only broadcasts within a warp.
    // Let thread 0 write the final sum and sync.
    if (tid == 0) {
        shared_reduction_buffer[0] = val;
    }
     __syncthreads(); // Ensure final sum is written
     val = shared_reduction_buffer[0]; // All threads read the final sum
     __syncthreads(); // Ensure all threads have read before shared buffer is reused

    return val;
}


// --- Parallel Block-Wide Exclusive Scan (Single Array - Adapted Harris) ---
// Input/Output: Modifies shared_data in place. Size = num_items <= BLOCK_SIZE.
// Uses BLOCK_SIZE threads. Assumes BLOCK_SIZE is power of 2.
template <typename scalar_t, int BLOCK_SIZE, int T_TILE_SIZE>
__device__ __forceinline__ void parallel_block_exclusive_scan(
    scalar_t* shared_data, // Array of size T_TILE_SIZE in shared memory
    scalar_t* temp_storage // Temporary storage, ideally size BLOCK_SIZE for intermediate steps if needed by algorithm, here used conceptually
    ) {
    // This version targets scanning an array of size T_TILE_SIZE using potentially more threads (BLOCK_SIZE).
    // Assumes T_TILE_SIZE is power of 2 and BLOCK_SIZE >= T_TILE_SIZE.
    // Only threads with tid < T_TILE_SIZE actively participate in modifying shared_data elements.
    // All threads might participate in intermediate steps if algorithm requires (like CUB BlockScan).
    // The Harris scan variant below primarily uses threads tid < T_TILE_SIZE.

    int tid = threadIdx.x;

    // Load data into registers (only relevant threads)
    scalar_t val = (tid < T_TILE_SIZE) ? shared_data[tid] : static_cast<scalar_t>(0.0);

    // Upsweep (reduction) phase - build sum tree in place
    for (int offset = 1; offset < T_TILE_SIZE; offset *= 2) {
        __syncthreads(); // Sync before reading potentially updated values
        scalar_t other = (tid >= offset && tid < T_TILE_SIZE) ? shared_data[tid - offset] : static_cast<scalar_t>(0.0);
        if (tid >= offset && tid < T_TILE_SIZE) {
            val = val + other; // Store sum in local register
        }
        __syncthreads(); // Sync before writing back potentially updated values
        if (tid >= offset && tid < T_TILE_SIZE) {
            shared_data[tid] = val; // Write back sum
        }
    }

    // Clear last element for exclusive scan property
    if (tid == T_TILE_SIZE - 1) { // Let the last active thread clear its element
        // Optional: store total sum here if needed: total_sum = val;
        shared_data[tid] = static_cast<scalar_t>(0.0);
    }
    __syncthreads(); // Ensure last element is cleared before downsweep

    // Downsweep phase - construct scan from sums
    for (int offset = T_TILE_SIZE / 2; offset > 0; offset /= 2) {
        __syncthreads(); // Sync before reading potentially updated values
        scalar_t left_val = (tid >= offset && tid < T_TILE_SIZE) ? shared_data[tid - offset] : static_cast<scalar_t>(0.0);
        scalar_t current_val = (tid >= offset && tid < T_TILE_SIZE) ? shared_data[tid] : static_cast<scalar_t>(0.0); // Val at shared_data[tid] before overwrite

        __syncthreads(); // Ensure all reads complete before writes

        if (tid >= offset && tid < T_TILE_SIZE) {
             shared_data[tid] = left_val;         // Write prefix sum from left neighbor
             shared_data[tid - offset] = current_val + left_val; // Update left neighbor (pass sum down)
        }
    }
    __syncthreads(); // Final sync to ensure all writes are complete
}


// --- Parallel Scan applied Element-wise to State Matrices ---
// Uses the parallel_block_exclusive_scan helper.
// Corrected loading pattern.
template <typename scalar_t, int BLOCK_SIZE, int T_TILE_SIZE>
__device__ __forceinline__ void parallel_block_exclusive_scan_state_v2(
    scalar_t* tile_k_sum_state, // Shared Mem: (T_TILE_SIZE, Df_padded)
    scalar_t* tile_kv_state,    // Shared Mem: (T_TILE_SIZE, Df, Dv_padded)
    int Df, int Dv, int Df_padded, int Dv_padded, // Original and padded dimensions
    scalar_t* temp_scan_buffer) // Temporary Shared Mem: (T_TILE_SIZE) elements
{
    int tid = threadIdx.x;
    const int num_k_scans = Df;
    const int num_kv_scans = Df * Dv;
    const int total_scans = num_k_scans + num_kv_scans;

    // Check requirement for parallel scan implementation
    // Host should already check this, but good practice for device code too.
    #if !defined(__CUDACC_RTC__) // Cannot use static_assert in NVRTC kernels easily
        static_assert(BLOCK_SIZE >= T_TILE_SIZE, "Parallel scan requires BLOCK_SIZE >= T_TILE_SIZE");
        static_assert((T_TILE_SIZE > 0) && ((T_TILE_SIZE & (T_TILE_SIZE - 1)) == 0), "T_TILE_SIZE must be a power of 2 for this scan");
    #endif

    // Parallelize scans over the Df/Dv dimensions using block stride loop
    for (int scan_idx = tid; scan_idx < total_scans; scan_idx += BLOCK_SIZE) {
        bool is_k_scan = scan_idx < num_k_scans;
        int d_idx = is_k_scan ? scan_idx : (scan_idx - num_k_scans); // Index within the specific state matrix column/vector

        // --- Load column/vector into temp buffer ---
        // Parallelize loading using all threads in the block with stride loop
        // Each thread `t_load` loads one element of the vector to be scanned.
        for (int t_load = tid; t_load < T_TILE_SIZE; t_load += BLOCK_SIZE) {
             if (is_k_scan) {
                 temp_scan_buffer[t_load] = tile_k_sum_state[t_load * Df_padded + d_idx];
             } else {
                 int df_kv = d_idx / Dv; int dv_kv = d_idx % Dv;
                 // Check layout: tile_kv_state is [T_TILE_SIZE, Df, Dv_padded]
                 temp_scan_buffer[t_load] = tile_kv_state[t_load * Df * Dv_padded + df_kv * Dv_padded + dv_kv];
             }
        }
        __syncthreads(); // Ensure load complete for this scan_idx

        // --- Perform scan on temp buffer using the parallel scan helper ---
        // Only need T_TILE_SIZE threads, but the helper manages participation.
        parallel_block_exclusive_scan<scalar_t, BLOCK_SIZE, T_TILE_SIZE>(temp_scan_buffer, nullptr); // Pass nullptr for temp_storage if unused by this impl.
        // __syncthreads() is called inside parallel_block_exclusive_scan

        // --- Write results back ---
        // Parallelize writing using all threads in the block with stride loop
        for (int t_write = tid; t_write < T_TILE_SIZE; t_write += BLOCK_SIZE) {
             if (is_k_scan) {
                  tile_k_sum_state[t_write * Df_padded + d_idx] = temp_scan_buffer[t_write];
             } else {
                 int df_kv = d_idx / Dv; int dv_kv = d_idx % Dv;
                 tile_kv_state[t_write * Df * Dv_padded + df_kv * Dv_padded + dv_kv] = temp_scan_buffer[t_write];
             }
        }
        __syncthreads(); // Ensure writes complete before next scan_idx iteration
    } // end loop over scans
    // No final syncthreads strictly needed here as the loop involves syncs
}


//----------------------------------------------------------------------------
// Kernel 1: Intra-Tile Processing, Parallel Scan, Summary Calculation (Optimized)
//----------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE, int T_TILE_SIZE>
__global__ void fa_kla_intra_tile_kernel(
    const scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const scalar_t* __restrict__ temp_ptr,
    scalar_t* __restrict__ tile_k_summaries,     // Output: [B, H, Df, NumTiles]
    scalar_t* __restrict__ tile_kv_summaries,    // Output: [B, H, Df, Dv, NumTiles]
    int B, int T, int H, int Df, int Dv,
    int k_stride_B, int k_stride_H, int k_stride_T,
    int v_stride_B, int v_stride_H, int v_stride_T,
    int temp_stride_B, int temp_stride_H, int temp_stride_T, // Added temp_stride_T
    // Strides for summaries (Corrected based on layout)
    int k_summary_stride_B, int k_summary_stride_H, int k_summary_stride_Df, int k_summary_stride_Tile,
    int kv_summary_stride_B, int kv_summary_stride_H, int kv_summary_stride_Df, int kv_summary_stride_Dv, int kv_summary_stride_Tile
)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tile_idx = blockIdx.z; // Index over NumTiles dimension

    const int tid = threadIdx.x;
    const int block_dim = blockDim.x; // Should be BLOCK_SIZE

    const int t_start = tile_idx * T_TILE_SIZE;
    const int t_end = min(t_start + T_TILE_SIZE, T);
    const int current_tile_len = t_end - t_start; // Number of valid time steps in this tile

    // Shared Memory Allocation (Using reinterpret_cast)
    const int Df_padded = get_padded_dim(Df, SHARED_MEM_ALIGNMENT);
    const int Dv_padded = get_padded_dim(Dv, SHARED_MEM_ALIGNMENT);

    // Layout:
    // shared_k_sum_local: [T_TILE_SIZE][Df_padded]
    // shared_kv_local:    [T_TILE_SIZE][Df][Dv_padded]
    // temp_scan_buffer:   [T_TILE_SIZE]
    extern __shared__ unsigned char shared_mem_raw[];
    scalar_t* shared_k_sum_local = reinterpret_cast<scalar_t*>(shared_mem_raw);
    scalar_t* shared_kv_local = shared_k_sum_local + T_TILE_SIZE * Df_padded;
    scalar_t* temp_scan_buffer = shared_kv_local + T_TILE_SIZE * Df * Dv_padded;

    // Base pointers for global memory access
    const scalar_t* base_k_ptr = k_ptr + b * k_stride_B + h * k_stride_H;
    const scalar_t* base_v_ptr = v_ptr + b * v_stride_B + h * v_stride_H;
    // temp layout is (B, H, T)
    const scalar_t* base_temp_ptr = temp_ptr + b * temp_stride_B + h * temp_stride_H;

    // Base pointers for output summaries (Corrected indexing below)
    scalar_t* base_k_summary_ptr = tile_k_summaries + b * k_summary_stride_B + h * k_summary_stride_H;
    scalar_t* base_kv_summary_ptr = tile_kv_summaries + b * kv_summary_stride_B + h * kv_summary_stride_H;


    // --- Load Tile Data into Shared Memory ---
    // Vectorization check (remains same)
    bool use_vec4_df = (Df % VEC_SIZE == 0) && std::is_same<scalar_t, float>::value;
    //bool use_vec4_dv = (Dv % VEC_SIZE == 0) && std::is_same<scalar_t, float>::value; // Not used in current kv loading

    // Load k_sum = k * temp
    // Parallelize over (t_local, df_idx) using block stride
    for (int load_idx = tid; load_idx < T_TILE_SIZE * Df; load_idx += block_dim) {
        int t_local = load_idx / Df;
        int df_idx = load_idx % Df;
        int t_global = t_start + t_local;
        scalar_t* smem_k_ptr = shared_k_sum_local + t_local * Df_padded + df_idx;

        if (t_global < T) {
            const scalar_t temp_val = base_temp_ptr[t_global * temp_stride_T]; // Use temp_stride_T
            const scalar_t* k_t_ptr = base_k_ptr + t_global * k_stride_T;
            *smem_k_ptr = k_t_ptr[df_idx] * temp_val;
        } else {
            *smem_k_ptr = static_cast<scalar_t>(0.0);
        }
        // Vectorization could be added here if beneficial, but strided access complicates it
    }
    // Zero out padding in shared_k_sum_local if needed (can be skipped if padded area is never read)
    for (int pad_idx = tid; pad_idx < T_TILE_SIZE * (Df_padded - Df); pad_idx += block_dim) {
         int t_local = pad_idx / (Df_padded - Df);
         int df_pad_offset = pad_idx % (Df_padded - Df);
         shared_k_sum_local[t_local * Df_padded + Df + df_pad_offset] = static_cast<scalar_t>(0.0);
    }


    // Load kv = outer(k*temp, v) -> k_t * temp_t * v_t^T
    // Parallelize over (t_local, df_idx, dv_idx) using block stride
    for (int load_idx = tid; load_idx < T_TILE_SIZE * Df * Dv; load_idx += block_dim) {
        int t_local = load_idx / (Df * Dv);
        int df_dv_idx = load_idx % (Df * Dv);
        int df_idx = df_dv_idx / Dv;
        int dv_idx = df_dv_idx % Dv;
        int t_global = t_start + t_local;

        // Destination: shared_kv_local[t_local][df_idx][dv_idx] (with Dv padding)
        scalar_t* smem_kv_ptr = shared_kv_local + t_local * Df * Dv_padded + df_idx * Dv_padded + dv_idx;

        if (t_global < T) {
            const scalar_t temp_val = base_temp_ptr[t_global * temp_stride_T]; // Use temp_stride_T
            const scalar_t* k_t_ptr = base_k_ptr + t_global * k_stride_T;
            const scalar_t* v_t_ptr = base_v_ptr + t_global * v_stride_T;
            *smem_kv_ptr = k_t_ptr[df_idx] * temp_val * v_t_ptr[dv_idx];
        } else {
            *smem_kv_ptr = static_cast<scalar_t>(0.0);
        }
    }
     // Zero out padding in shared_kv_local if needed
    for (int pad_idx = tid; pad_idx < T_TILE_SIZE * Df * (Dv_padded - Dv); pad_idx += block_dim) {
         int t_local = pad_idx / (Df * (Dv_padded - Dv));
         int df_pad_dv_offset = pad_idx % (Df * (Dv_padded - Dv));
         int df_idx = df_pad_dv_offset / (Dv_padded - Dv);
         int dv_pad_offset = df_pad_dv_offset % (Dv_padded - Dv);
         shared_kv_local[t_local * Df * Dv_padded + df_idx * Dv_padded + Dv + dv_pad_offset] = static_cast<scalar_t>(0.0);
    }

    __syncthreads(); // Ensure all loads and padding zeros are complete

    // --- Perform Parallel Intra-Tile Exclusive Scan ---
    // Check done in helper, assumes BLOCK_SIZE >= T_TILE_SIZE and T_TILE_SIZE is power-of-2
    parallel_block_exclusive_scan_state_v2<scalar_t, BLOCK_SIZE, T_TILE_SIZE>(
        shared_k_sum_local, shared_kv_local,
        Df, Dv, Df_padded, Dv_padded,
        temp_scan_buffer);
    // Result: shared_*_local now contain exclusive prefix sums within the tile.
    // __syncthreads() happens inside scan helper


    // --- Calculate and Store Tile Summaries (Corrected Indexing and Logic) ---
    // The summary is the inclusive sum up to the end of the tile.
    // It's the exclusive sum of the *next* element (if it existed),
    // which equals exclusive_sum[last_element] + value[last_element].

    if (current_tile_len > 0) {
        int t_last_local = current_tile_len - 1;
        int t_last_global = t_start + t_last_local; // t_end - 1

        // Calculate k summaries [B, H, Df, NumTiles]
        // Parallelize over Df using block stride
        for (int df_idx = tid; df_idx < Df; df_idx += block_dim) {
            scalar_t local_scan_last = shared_k_sum_local[t_last_local * Df_padded + df_idx]; // Exclusive sum result for last element
            // Read original values for the last element again from global memory
            scalar_t last_k_val = base_k_ptr[t_last_global * k_stride_T + df_idx];
            scalar_t last_temp_val = base_temp_ptr[t_last_global * temp_stride_T];
            scalar_t last_element_val = last_k_val * last_temp_val;
            // Correct summary calculation: exclusive_sum[last] + value[last] = inclusive_sum[last]
            scalar_t k_summary_val = local_scan_last + last_element_val;

            // Corrected Indexing: base + df * stride_Df + tile * stride_Tile
            base_k_summary_ptr[df_idx * k_summary_stride_Df + tile_idx * k_summary_stride_Tile] = k_summary_val;
        }

        // Calculate kv summaries [B, H, Df, Dv, NumTiles]
        // Parallelize over Df*Dv using block stride
        for (int kv_idx = tid; kv_idx < Df * Dv; kv_idx += block_dim) {
             int df_idx = kv_idx / Dv;
             int dv_idx = kv_idx % Dv;

             scalar_t local_scan_last = shared_kv_local[t_last_local * Df * Dv_padded + df_idx * Dv_padded + dv_idx]; // Exclusive sum result
             // Read original values for the last element again
             scalar_t last_k_val = base_k_ptr[t_last_global * k_stride_T + df_idx];
             scalar_t last_v_val = base_v_ptr[t_last_global * v_stride_T + dv_idx];
             scalar_t last_temp_val = base_temp_ptr[t_last_global * temp_stride_T];
             scalar_t last_element_val = last_k_val * last_temp_val * last_v_val;
             // Correct summary calculation
             scalar_t kv_summary_val = local_scan_last + last_element_val;

             // Corrected Indexing: base + df*stride_Df + dv*stride_Dv + tile*stride_Tile
             base_kv_summary_ptr[df_idx * kv_summary_stride_Df + dv_idx * kv_summary_stride_Dv + tile_idx * kv_summary_stride_Tile] = kv_summary_val;
        }
    } else {
        // Handle empty tiles (e.g., if T is multiple of T_TILE_SIZE and NumTiles > T/T_TILE_SIZE)
        // Write zero summaries
         for (int df_idx = tid; df_idx < Df; df_idx += block_dim) {
             base_k_summary_ptr[df_idx * k_summary_stride_Df + tile_idx * k_summary_stride_Tile] = static_cast<scalar_t>(0.0);
         }
         for (int kv_idx = tid; kv_idx < Df * Dv; kv_idx += block_dim) {
             int df_idx = kv_idx / Dv; int dv_idx = kv_idx % Dv;
             base_kv_summary_ptr[df_idx * kv_summary_stride_Df + dv_idx * kv_summary_stride_Dv + tile_idx * kv_summary_stride_Tile] = static_cast<scalar_t>(0.0);
         }
    }
    // No final syncthreads needed as writes are independent per thread after calculation
}


//----------------------------------------------------------------------------
// Kernel 3: Correction and Final Output Calculation (Optimized Reduction & Scan)
//----------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE, int T_TILE_SIZE>
__global__ void fa_kla_correction_kernel(
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const scalar_t* __restrict__ temp_ptr,
    const scalar_t* __restrict__ tile_k_prefixes,  // Input: [B, H, Df, NumTiles]
    const scalar_t* __restrict__ tile_kv_prefixes, // Input: [B, H, Df, Dv, NumTiles]
    scalar_t* __restrict__ out_ptr,               // Output: [B, H, T, Dv]
    int B, int T, int H, int Df, int Dv,
    int q_stride_B, int q_stride_H, int q_stride_T,
    int k_stride_B, int k_stride_H, int k_stride_T,
    int v_stride_B, int v_stride_H, int v_stride_T,
    int temp_stride_B, int temp_stride_H, int temp_stride_T, // Added temp_stride_T
    // Strides for prefixes (Corrected based on layout)
    int k_prefix_stride_B, int k_prefix_stride_H, int k_prefix_stride_Df, int k_prefix_stride_Tile,
    int kv_prefix_stride_B, int kv_prefix_stride_H, int kv_prefix_stride_Df, int kv_prefix_stride_Dv, int kv_prefix_stride_Tile,
    int out_stride_B, int out_stride_H, int out_stride_T,
    float epsilon // Pass float epsilon for consistency, cast inside
)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tile_idx = blockIdx.z; // Index over NumTiles dimension

    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;

    const int t_start = tile_idx * T_TILE_SIZE;
    const int t_end = min(t_start + T_TILE_SIZE, T);
    const int current_tile_len = t_end - t_start;

    // Shared Memory Allocation (Using reinterpret_cast)
    const int Df_padded = get_padded_dim(Df, SHARED_MEM_ALIGNMENT);
    const int Dv_padded = get_padded_dim(Dv, SHARED_MEM_ALIGNMENT);
    constexpr int REDUCTION_BUFFER_SIZE = BLOCK_SIZE / warpSize; // Size based on block_reduce_sum

    // Layout:
    // shared_tile_k_sum_scan: [T_TILE_SIZE][Df_padded] (Result of intra-tile scan)
    // shared_tile_kv_scan:    [T_TILE_SIZE][Df][Dv_padded] (Result of intra-tile scan)
    // shared_prefix_k:        [Df] (Prefix sum from previous tiles)
    // shared_prefix_kv:       [Df * Dv] (Prefix sum from previous tiles)
    // shared_reduction_buffer:[REDUCTION_BUFFER_SIZE]
    // temp_scan_buffer:       [T_TILE_SIZE]
    extern __shared__ unsigned char shared_mem_raw[];
    scalar_t* shared_tile_k_sum_scan = reinterpret_cast<scalar_t*>(shared_mem_raw);
    scalar_t* shared_tile_kv_scan = shared_tile_k_sum_scan + T_TILE_SIZE * Df_padded;
    // Careful: Df*Dv for prefix might be large. Ensure shared mem is sufficient.
    scalar_t* shared_prefix_k = shared_tile_kv_scan + T_TILE_SIZE * Df * Dv_padded;
    scalar_t* shared_prefix_kv = shared_prefix_k + Df; // Store as [Df][Dv] effectively
    scalar_t* shared_reduction_buffer = shared_prefix_kv + Df * Dv;
    scalar_t* temp_scan_buffer = shared_reduction_buffer + REDUCTION_BUFFER_SIZE;


    // --- Base Pointers ---
    const scalar_t* base_q_ptr = q_ptr + b * q_stride_B + h * q_stride_H;
    const scalar_t* base_k_ptr = k_ptr + b * k_stride_B + h * k_stride_H;
    const scalar_t* base_v_ptr = v_ptr + b * v_stride_B + h * v_stride_H;
    const scalar_t* base_temp_ptr = temp_ptr + b * temp_stride_B + h * temp_stride_H;
    scalar_t* base_out_ptr = out_ptr + b * out_stride_B + h * out_stride_H;

    // Base pointers for input prefixes (Corrected indexing below)
    const scalar_t* base_k_prefix_ptr = tile_k_prefixes + b * k_prefix_stride_B + h * k_prefix_stride_H;
    const scalar_t* base_kv_prefix_ptr = tile_kv_prefixes + b * kv_prefix_stride_B + h * kv_prefix_stride_H;

    // --- Load Tile Prefix Sum into Shared Memory ---
    // Parallelize loading over Df and Df*Dv using block stride
    if (tile_idx > 0) {
        // Load K prefixes
        for (int df_idx = tid; df_idx < Df; df_idx += block_dim) {
            // Corrected Indexing: base + df*stride_Df + tile*stride_Tile
            shared_prefix_k[df_idx] = base_k_prefix_ptr[df_idx * k_prefix_stride_Df + tile_idx * k_prefix_stride_Tile];
        }
        // Load KV prefixes
        for (int kv_idx = tid; kv_idx < Df * Dv; kv_idx += block_dim) {
            int df_idx = kv_idx / Dv;
            int dv_idx = kv_idx % Dv;
            // Corrected Indexing: base + df*stride_Df + dv*stride_Dv + tile*stride_Tile
            shared_prefix_kv[kv_idx] = base_kv_prefix_ptr[df_idx * kv_prefix_stride_Df + dv_idx * kv_prefix_stride_Dv + tile_idx * kv_prefix_stride_Tile];
        }
    } else { // Tile 0 has zero prefix sum
        for (int df_idx = tid; df_idx < Df; df_idx += block_dim) {
            shared_prefix_k[df_idx] = static_cast<scalar_t>(0.0);
        }
        for (int kv_idx = tid; kv_idx < Df * Dv; kv_idx += block_dim) {
            shared_prefix_kv[kv_idx] = static_cast<scalar_t>(0.0);
        }
    }
    // No syncthreads needed yet, loading is independent.


    // --- Recompute Intra-Tile Scan (Load & Scan) ---
    // Re-load k_sum = k * temp into shared_tile_k_sum_scan (padded)
    for (int load_idx = tid; load_idx < T_TILE_SIZE * Df; load_idx += block_dim) {
        int t_local = load_idx / Df;
        int df_idx = load_idx % Df;
        int t_global = t_start + t_local;
        scalar_t* smem_k_ptr = shared_tile_k_sum_scan + t_local * Df_padded + df_idx;
        if (t_global < T) {
            const scalar_t temp_val = base_temp_ptr[t_global * temp_stride_T];
            const scalar_t* k_t_ptr = base_k_ptr + t_global * k_stride_T;
            *smem_k_ptr = k_t_ptr[df_idx] * temp_val;
        } else {
            *smem_k_ptr = static_cast<scalar_t>(0.0);
        }
    }
    // Zero padding
    for (int pad_idx = tid; pad_idx < T_TILE_SIZE * (Df_padded - Df); pad_idx += block_dim) {
         int t_local = pad_idx / (Df_padded - Df); int df_pad_offset = pad_idx % (Df_padded - Df);
         shared_tile_k_sum_scan[t_local * Df_padded + Df + df_pad_offset] = static_cast<scalar_t>(0.0);
    }

    // Re-load kv = outer(k*temp, v) into shared_tile_kv_scan (padded)
    for (int load_idx = tid; load_idx < T_TILE_SIZE * Df * Dv; load_idx += block_dim) {
        int t_local = load_idx / (Df * Dv); int df_dv_idx = load_idx % (Df * Dv);
        int df_idx = df_dv_idx / Dv; int dv_idx = df_dv_idx % Dv;
        int t_global = t_start + t_local;
        scalar_t* smem_kv_ptr = shared_tile_kv_scan + t_local * Df * Dv_padded + df_idx * Dv_padded + dv_idx;
        if (t_global < T) {
            const scalar_t temp_val = base_temp_ptr[t_global * temp_stride_T];
            const scalar_t* k_t_ptr = base_k_ptr + t_global * k_stride_T; const scalar_t* v_t_ptr = base_v_ptr + t_global * v_stride_T;
            *smem_kv_ptr = k_t_ptr[df_idx] * temp_val * v_t_ptr[dv_idx];
        } else {
            *smem_kv_ptr = static_cast<scalar_t>(0.0);
        }
    }
    // Zero padding
    for (int pad_idx = tid; pad_idx < T_TILE_SIZE * Df * (Dv_padded - Dv); pad_idx += block_dim) {
         int t_local = pad_idx / (Df * (Dv_padded - Dv)); int df_pad_dv_offset = pad_idx % (Df * (Dv_padded - Dv));
         int df_idx = df_pad_dv_offset / (Dv_padded - Dv); int dv_pad_offset = df_pad_dv_offset % (Dv_padded - Dv);
         shared_tile_kv_scan[t_local * Df * Dv_padded + df_idx * Dv_padded + Dv + dv_pad_offset] = static_cast<scalar_t>(0.0);
    }

    __syncthreads(); // Sync AFTER loading prefixes AND tile data

    // Perform parallel exclusive scan on shared memory state
    parallel_block_exclusive_scan_state_v2<scalar_t, BLOCK_SIZE, T_TILE_SIZE>(
        shared_tile_k_sum_scan, shared_tile_kv_scan,
        Df, Dv, Df_padded, Dv_padded,
        temp_scan_buffer);
    // Scan results are now in shared_tile_*_scan
    // __syncthreads() happens inside scan helper

    // --- Compute Final Output ---
    // Use type-dependent epsilon cast to scalar_t
    const scalar_t stable_eps = static_cast<scalar_t>(epsilon);

    // Loop over time steps within the tile handled by this block
    for (int t_local = 0; t_local < current_tile_len; ++t_local) {
        int t_global = t_start + t_local;

        const scalar_t* q_t_ptr = base_q_ptr + t_global * q_stride_T;
        scalar_t* out_t_ptr = base_out_ptr + t_global * out_stride_T;
        const scalar_t* k_t_ptr = base_k_ptr + t_global * k_stride_T;
        const scalar_t* v_t_ptr = base_v_ptr + t_global * v_stride_T;
        const scalar_t temp_val_t = base_temp_ptr[t_global * temp_stride_T];

        // --- Calculate Denominator: q_t @ K_sum_corrected ---
        // K_sum_corrected = prefix_k + intra_tile_exclusive_scan_k + current_k*temp
        scalar_t thread_den_sum = static_cast<scalar_t>(0.0);
        // Parallel reduction over Df dimension
        for (int df_idx = tid; df_idx < Df; df_idx += block_dim) {
            scalar_t k_sum_local_scan = shared_tile_k_sum_scan[t_local * Df_padded + df_idx]; // Excl scan result up to t_local
            scalar_t k_val = k_t_ptr[df_idx];
            scalar_t k_current_term = k_val * temp_val_t; // Value at current time step t
            scalar_t k_sum_corrected = shared_prefix_k[df_idx] + k_sum_local_scan + k_current_term; // Inclusive sum up to t
            thread_den_sum += q_t_ptr[df_idx] * k_sum_corrected;
        }
        // Reduce across the block
        scalar_t den_t = block_reduce_sum<scalar_t, BLOCK_SIZE>(thread_den_sum, shared_reduction_buffer);
        // __syncthreads() happens inside block_reduce_sum

        // --- Calculate Numerator: q_t @ KV_sum_corrected ---
        // KV_sum_corrected = prefix_kv + intra_tile_exclusive_scan_kv + current_kv_outer
        // Result is a vector of size Dv. Parallelize calculation over Dv.
        for (int dv_idx = tid; dv_idx < Dv; dv_idx += block_dim) {
            scalar_t num_val = static_cast<scalar_t>(0.0);
            // Inner loop over Df (reduction dimension)
            for (int df_idx = 0; df_idx < Df; ++df_idx) {
                 int kv_scan_idx = t_local * Df * Dv_padded + df_idx * Dv_padded + dv_idx;
                 int kv_prefix_idx = df_idx * Dv + dv_idx; // Index into flat shared_prefix_kv

                 scalar_t kv_sum_local_scan = shared_tile_kv_scan[kv_scan_idx]; // Excl scan result
                 scalar_t k_val = k_t_ptr[df_idx];
                 scalar_t v_val = v_t_ptr[dv_idx];
                 scalar_t kv_outer_current = k_val * temp_val_t * v_val; // Value at current time step t

                 scalar_t kv_sum_corrected = shared_prefix_kv[kv_prefix_idx] + kv_sum_local_scan + kv_outer_current; // Inclusive sum up to t
                 num_val += q_t_ptr[df_idx] * kv_sum_corrected;
            }
            // Store numerator result before division
            // Use atomicAdd if multiple threads could write to the same out_t_ptr[dv_idx]?
            // No, the outer loop `for (int dv_idx = tid; ...)` ensures each thread writes to a unique dv_idx.
            out_t_ptr[dv_idx] = num_val;
        }
        __syncthreads(); // Ensure all numerator calculations and writes are complete before division

        // --- Final Division ---
        // Note: den_t already includes the contribution from the current time step t
        scalar_t scale = static_cast<scalar_t>(1.0) / (den_t + stable_eps);
        for (int dv_idx = tid; dv_idx < Dv; dv_idx += block_dim) {
            out_t_ptr[dv_idx] *= scale;
        }
        __syncthreads(); // Sync before processing the next t_local to ensure writes are visible if needed later (e.g., debugging)
                        // and prevent race conditions if shared memory is reused immediately.
    } // End loop over t_local
}


//----------------------------------------------------------------------------
// Host-Side Function Implementation (Corrected CUB Scan & Launch)
//----------------------------------------------------------------------------
torch::Tensor fa_kla_attention_cuda(
    const torch::Tensor& q_mapped,
    const torch::Tensor& k_mapped,
    const torch::Tensor& v_reshaped,
    const torch::Tensor& temp)
{
    // --- Validation ---
    TORCH_CHECK(q_mapped.is_cuda() && k_mapped.is_cuda() && v_reshaped.is_cuda() && temp.is_cuda(), "Inputs must be CUDA tensors");
    const auto dtype = q_mapped.scalar_type();
    TORCH_CHECK(k_mapped.scalar_type() == dtype, "k dtype mismatch");
    TORCH_CHECK(v_reshaped.scalar_type() == dtype, "v dtype mismatch");
    TORCH_CHECK(temp.scalar_type() == dtype, "temp dtype mismatch");

    TORCH_CHECK(q_mapped.dim() == 4, "q must be 4D (B, H, T, Df)"); // Assuming layout (B, H, T, Df)
    TORCH_CHECK(k_mapped.dim() == 4, "k must be 4D (B, H, T, Df)"); // Assuming layout (B, H, T, Df)
    TORCH_CHECK(v_reshaped.dim() == 4, "v must be 4D (B, H, T, Dv)"); // Assuming layout (B, H, T, Dv)
    TORCH_CHECK(temp.dim() == 3, "temp must be 3D (B, H, T)");    // Assuming layout (B, H, T)

    TORCH_CHECK(q_mapped.size(0) == k_mapped.size(0) && q_mapped.size(0) == v_reshaped.size(0) && q_mapped.size(0) == temp.size(0), "Batch size mismatch");
    TORCH_CHECK(q_mapped.size(1) == k_mapped.size(1) && q_mapped.size(1) == v_reshaped.size(1) && q_mapped.size(1) == temp.size(1), "Head count mismatch");
    TORCH_CHECK(q_mapped.size(2) == k_mapped.size(2) && q_mapped.size(2) == v_reshaped.size(2) && q_mapped.size(2) == temp.size(2), "Sequence length mismatch");
    TORCH_CHECK(q_mapped.size(3) == k_mapped.size(3), "Df dimension mismatch");

    // Enforce contiguity for performance and kernel assumptions
    // Note: contiguous() might copy data if not already contiguous.
    auto q_c = q_mapped.contiguous();
    auto k_c = k_mapped.contiguous();
    auto v_c = v_reshaped.contiguous();
    auto temp_c = temp.contiguous();

    // --- Dimensions & Tiling ---
    const int B = q_c.size(0);
    const int H = q_c.size(1);
    const int T = q_c.size(2);
    const int Df = q_c.size(3);
    const int Dv = v_c.size(3);

    const int BLOCK_SIZE = FA_KLA_CUDA_BLOCK_SIZE;
    const int T_TILE_SIZE = FA_KLA_T_TILE_SIZE;
    TORCH_CHECK(T_TILE_SIZE <= BLOCK_SIZE, "T_TILE_SIZE must be <= BLOCK_SIZE for current parallel scan implementation");
    // Add check for power-of-2 if the scan requires it
    TORCH_CHECK((T_TILE_SIZE > 0) && ((T_TILE_SIZE & (T_TILE_SIZE - 1)) == 0), "T_TILE_SIZE must be a power of 2 for current scan implementation.");

    const int NumTiles = (T + T_TILE_SIZE - 1) / T_TILE_SIZE;

    // --- Allocate Buffers ---
    auto summary_options = q_c.options();
    // Layout: [B, H, Df, NumTiles] for k_summaries/prefixes
    auto tile_k_summaries = torch::empty({B, H, Df, NumTiles}, summary_options);
    TORCH_CHECK(tile_k_summaries.defined(), "Failed to allocate tile_k_summaries");
    // Layout: [B, H, Df, Dv, NumTiles] for kv_summaries/prefixes
    auto tile_kv_summaries = torch::empty({B, H, Df, Dv, NumTiles}, summary_options);
    TORCH_CHECK(tile_kv_summaries.defined(), "Failed to allocate tile_kv_summaries");

    // Prefixes will hold the exclusive scan result
    auto tile_k_prefixes = torch::empty_like(tile_k_summaries);
    TORCH_CHECK(tile_k_prefixes.defined(), "Failed to allocate tile_k_prefixes");
    auto tile_kv_prefixes = torch::empty_like(tile_kv_summaries);
    TORCH_CHECK(tile_kv_prefixes.defined(), "Failed to allocate tile_kv_prefixes");

    // Output tensor, same shape and layout as v_c potentially (B, H, T, Dv)
    auto output = torch::empty_like(v_c);
    TORCH_CHECK(output.defined(), "Failed to allocate output tensor");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- Kernel 1: Calculate Tile Summaries ---
    const dim3 grid1(B, H, NumTiles);
    const dim3 block1(BLOCK_SIZE);

    const int Df_padded = get_padded_dim(Df, SHARED_MEM_ALIGNMENT);
    const int Dv_padded = get_padded_dim(Dv, SHARED_MEM_ALIGNMENT);

    // Calculate Shared Memory for Kernel 1
    size_t shared_mem1_bytes = (
        (size_t)T_TILE_SIZE * Df_padded       // shared_k_sum_local
        + (size_t)T_TILE_SIZE * Df * Dv_padded  // shared_kv_local
        + (size_t)T_TILE_SIZE                 // temp_scan_buffer
    ) * torch::elementSize(dtype);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "fa_kla_intra_tile_launcher", [&] {
        // Strides for original inputs (assuming contiguous B, H, T, Dim)
        int k_stride_B = k_c.stride(0), k_stride_H = k_c.stride(1), k_stride_T = k_c.stride(2); // k_stride_Df = k_c.stride(3) = 1
        int v_stride_B = v_c.stride(0), v_stride_H = v_c.stride(1), v_stride_T = v_c.stride(2); // v_stride_Dv = v_c.stride(3) = 1
        int temp_stride_B = temp_c.stride(0), temp_stride_H = temp_c.stride(1), temp_stride_T = temp_c.stride(2); // temp_stride_T added

        // Strides for summary outputs (Layout: B, H, Df, NumTiles) & (B, H, Df, Dv, NumTiles)
        int k_summary_stride_B = tile_k_summaries.stride(0);
        int k_summary_stride_H = tile_k_summaries.stride(1);
        int k_summary_stride_Df = tile_k_summaries.stride(2); // Stride between Df elements
        int k_summary_stride_Tile = tile_k_summaries.stride(3); // Stride between NumTiles elements (should be 1)

        int kv_summary_stride_B = tile_kv_summaries.stride(0);
        int kv_summary_stride_H = tile_kv_summaries.stride(1);
        int kv_summary_stride_Df = tile_kv_summaries.stride(2); // Stride between Df elements
        int kv_summary_stride_Dv = tile_kv_summaries.stride(3); // Stride between Dv elements
        int kv_summary_stride_Tile = tile_kv_summaries.stride(4); // Stride between NumTiles elements (should be 1)

        fa_kla_intra_tile_kernel<scalar_t, BLOCK_SIZE, T_TILE_SIZE><<<grid1, block1, shared_mem1_bytes, stream>>>(
            k_c.data_ptr<scalar_t>(),
            v_c.data_ptr<scalar_t>(),
            temp_c.data_ptr<scalar_t>(),
            tile_k_summaries.data_ptr<scalar_t>(),
            tile_kv_summaries.data_ptr<scalar_t>(),
            B, T, H, Df, Dv,
            k_stride_B, k_stride_H, k_stride_T,
            v_stride_B, v_stride_H, v_stride_T,
            temp_stride_B, temp_stride_H, temp_stride_T, // Pass temp_stride_T
            k_summary_stride_B, k_summary_stride_H, k_summary_stride_Df, k_summary_stride_Tile,
            kv_summary_stride_B, kv_summary_stride_H, kv_summary_stride_Df, kv_summary_stride_Dv, kv_summary_stride_Tile
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    // --- Kernel 2: Perform Exclusive Scan on Summaries using CUB ---
    // CUB expects data where the dimension to be scanned is contiguous for each segment.
    // k_summaries: [B, H, Df, NumTiles] - Scan over NumTiles. Contiguous.
    // kv_summaries: [B, H, Df, Dv, NumTiles] - Scan over NumTiles. Contiguous.

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    const int num_k_segments = B * H * Df;       // Each (b, h, df) is an independent scan
    const int num_kv_segments = B * H * Df * Dv; // Each (b, h, df, dv) is an independent scan
    const int scan_length = NumTiles;            // Length of each scan

    // Use a temporary tensor for CUB storage
    at::Tensor temp_storage; // Defined outside dispatch

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "fa_kla_cub_scan", [&] {
        scalar_t* d_k_summaries_in = tile_k_summaries.data_ptr<scalar_t>();
        scalar_t* d_k_prefixes_out = tile_k_prefixes.data_ptr<scalar_t>();
        scalar_t* d_kv_summaries_in = tile_kv_summaries.data_ptr<scalar_t>();
        scalar_t* d_kv_prefixes_out = tile_kv_prefixes.data_ptr<scalar_t>();

        // First call to get storage size for K scan
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_k_summaries_in, d_k_prefixes_out, scan_length, num_k_segments, stream);

        // Allocate storage (once, potentially resizing for KV scan)
        temp_storage = torch::empty({(int64_t)temp_storage_bytes}, summary_options.dtype(torch::kByte));
        TORCH_CHECK(temp_storage.defined(), "Failed to allocate CUB temp storage for K scan");
        d_temp_storage = temp_storage.data_ptr();

        // Second call to perform K scan
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_k_summaries_in, d_k_prefixes_out, scan_length, num_k_segments, stream);

        // First call to get storage size for KV scan
        size_t temp_storage_bytes_kv = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes_kv, d_kv_summaries_in, d_kv_prefixes_out, scan_length, num_kv_segments, stream);

        // Check if KV scan requires more storage
        if (temp_storage_bytes_kv > temp_storage_bytes) {
             temp_storage = torch::empty({(int64_t)temp_storage_bytes_kv}, summary_options.dtype(torch::kByte));
             TORCH_CHECK(temp_storage.defined(), "Failed to allocate CUB temp storage for KV scan");
             d_temp_storage = temp_storage.data_ptr();
             temp_storage_bytes = temp_storage_bytes_kv; // Update size
        } else if (temp_storage_bytes_kv == 0 && temp_storage_bytes == 0) {
            // Handle case where both scans might need 0 bytes (unlikely but possible)
            d_temp_storage = nullptr; // Ensure temp storage pointer is null if size is 0
        }
        // else: Existing buffer is sufficient or larger

        // Second call to perform KV scan (using potentially resized buffer)
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_kv_summaries_in, d_kv_prefixes_out, scan_length, num_kv_segments, stream);
    });
    AT_CUDA_CHECK(cudaGetLastError());


    // --- Kernel 3: Correction and Final Output ---
    const dim3 grid3 = grid1; // Same grid as Kernel 1 (B, H, NumTiles)
    const dim3 block3 = block1; // Same block size

    // Calculate Shared Memory for Kernel 3
    constexpr int REDUCTION_BUFFER_SIZE = BLOCK_SIZE / 32; // Assuming warpSize is 32
    size_t shared_mem3_bytes = (
        (size_t)T_TILE_SIZE * Df_padded        // shared_tile_k_sum_scan
        + (size_t)T_TILE_SIZE * Df * Dv_padded   // shared_tile_kv_scan
        + (size_t)Df                           // shared_prefix_k
        + (size_t)Df * Dv                      // shared_prefix_kv
        + (size_t)REDUCTION_BUFFER_SIZE        // shared_reduction_buffer
        + (size_t)T_TILE_SIZE                  // temp_scan_buffer
    ) * torch::elementSize(dtype);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "fa_kla_correction_launcher", [&] {
        // Input Strides
        int q_stride_B = q_c.stride(0), q_stride_H = q_c.stride(1), q_stride_T = q_c.stride(2);
        int k_stride_B = k_c.stride(0), k_stride_H = k_c.stride(1), k_stride_T = k_c.stride(2);
        int v_stride_B = v_c.stride(0), v_stride_H = v_c.stride(1), v_stride_T = v_c.stride(2);
        int temp_stride_B = temp_c.stride(0), temp_stride_H = temp_c.stride(1), temp_stride_T = temp_c.stride(2);

        // Prefix Strides (Layouts: B, H, Df, NumTiles and B, H, Df, Dv, NumTiles)
        int k_prefix_stride_B = tile_k_prefixes.stride(0);
        int k_prefix_stride_H = tile_k_prefixes.stride(1);
        int k_prefix_stride_Df = tile_k_prefixes.stride(2);
        int k_prefix_stride_Tile = tile_k_prefixes.stride(3);

        int kv_prefix_stride_B = tile_kv_prefixes.stride(0);
        int kv_prefix_stride_H = tile_kv_prefixes.stride(1);
        int kv_prefix_stride_Df = tile_kv_prefixes.stride(2);
        int kv_prefix_stride_Dv = tile_kv_prefixes.stride(3);
        int kv_prefix_stride_Tile = tile_kv_prefixes.stride(4);

        // Output Strides (Layout: B, H, T, Dv)
        int out_stride_B = output.stride(0), out_stride_H = output.stride(1), out_stride_T = output.stride(2);

        // Determine type-dependent epsilon (passed as float)
        float epsilon_val = std::is_same<scalar_t, at::Half>::value ? 1e-6f : 1e-12f; // Use appropriate default

        fa_kla_correction_kernel<scalar_t, BLOCK_SIZE, T_TILE_SIZE><<<grid3, block3, shared_mem3_bytes, stream>>>(
            q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), temp_c.data_ptr<scalar_t>(),
            tile_k_prefixes.data_ptr<scalar_t>(), tile_kv_prefixes.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, T, H, Df, Dv,
            q_stride_B, q_stride_H, q_stride_T, k_stride_B, k_stride_H, k_stride_T,
            v_stride_B, v_stride_H, v_stride_T, temp_stride_B, temp_stride_H, temp_stride_T,
            k_prefix_stride_B, k_prefix_stride_H, k_prefix_stride_Df, k_prefix_stride_Tile,
            kv_prefix_stride_B, kv_prefix_stride_H, kv_prefix_stride_Df, kv_prefix_stride_Dv, kv_prefix_stride_Tile,
            out_stride_B, out_stride_H, out_stride_T,
            epsilon_val
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return output;
}

} 
} 
} 
} 
