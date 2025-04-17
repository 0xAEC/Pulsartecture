// ultra_rwka/backend/cuda/wavelets/kernel.cu

#include "kernel.cuh"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <vector>

// Configurable constants
constexpr int WAVELET_DWT_CUDA_BLOCK_SIZE = 256; // Threads per block
constexpr int MAX_WPD_LEVELS = 4;        // Max decomposition levels supported
constexpr int WAVELET_SHARED_MEM_PADDING = 16; // Padding alignment (in elements) for shared mem

// Vectorization
using scalar4 = float4;
constexpr int VEC_SIZE = 4; // For float4


namespace ultra_rwka {
namespace backend {
namespace cuda {
namespace wavelets {

// Helper for padding
__host__ __device__ inline int get_padded_dim_wavelet(int dim, int align) {
    // Ensure alignment is at least VEC_SIZE for vector loads/stores
    int vec_align = align > VEC_SIZE ? align : VEC_SIZE;
    return (dim + vec_align - 1) / vec_align * vec_align;
}

//----------------------------------------------------------------------------
// Device Helper: Parallel 1D Convolution (Shared Memory based - Vectorized)
//----------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE, bool USE_VEC = (std::is_same<scalar_t, float>::value)>
__device__ void parallel_convolve_1d_vec(
    const scalar_t* input,     // Input signal segment (SHARED MEMORY - PADDED)
    int input_len,             // Original length (before padding)
    int input_len_padded,      // Padded length (in shared memory)
    const scalar_t* filter,    // Filter coefficients (SHARED MEMORY - PADDED/ALIGNED)
    int filter_len,
    scalar_t* output,          // Output coefficients (SHARED MEMORY - PADDED/ALIGNED)
    int output_len)            // Convolution output length (input_len + filter_len - 1)
{
    int tid = threadIdx.x;

    // Check if vectorization is possible and enabled
    const bool can_vectorize = USE_VEC && (filter_len % VEC_SIZE == 0) && (input_len_padded % VEC_SIZE == 0);

    // Each thread computes one output element
    for (int k_out = tid; k_out < output_len; k_out += BLOCK_SIZE) {
        scalar_t conv_val = static_cast<scalar_t>(0.0);
        const scalar_t* current_filter_rev = filter; // Filter assumed pre-reversed or handled below

        if (can_vectorize) {
            // Vectorized dot product
            scalar4 acc4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int k_vec = 0; k_vec < filter_len / VEC_SIZE; ++k_vec) {
                 // Input index needs careful calculation relative to k_out and padding
                 // Convolution: output[n] = sum_k input[n+k] * filter_reversed[k]
                 // OR output[n] = sum_k input[n-k] * filter[k] (requires reversed filter read)
                 // Let's use the second form with reversed filter read.
                 int input_base_idx = k_out + (k_vec * VEC_SIZE); // Base index for input vector load
                 // Check bounds carefully based on padding strategy used during load
                 // Assuming symmetric padding reflected in shared_input array of size input_len_padded
                 if (input_base_idx >= 0 && input_base_idx < input_len_padded - (VEC_SIZE - 1)) {
                      scalar4 input_vec = reinterpret_cast<const scalar4*>(input)[input_base_idx / VEC_SIZE];
                      // Filter is loaded reversed? Assume filter is loaded normally. Read reversed.
                      scalar4 filter_vec = reinterpret_cast<const scalar4*>(filter)[(filter_len / VEC_SIZE - 1) - k_vec];

                      // Fused multiply-add (FMA) if available
                      acc4.x += input_vec.x * filter_vec.x;
                      acc4.y += input_vec.y * filter_vec.y;
                      acc4.z += input_vec.z * filter_vec.z;
                      acc4.w += input_vec.w * filter_vec.w;
                 }
                 // TODO: Handle boundary conditions more precisely if needed
            }
            // Horizontal sum of accumulator vector
            conv_val = acc4.x + acc4.y + acc4.z + acc4.w;

            // Handle remainder if filter_len is not divisible by VEC_SIZE (though check prevents this path now)
            // for (int k = (filter_len / VEC_SIZE) * VEC_SIZE; k < filter_len; ++k) { ... scalar part ... }

        } else { // Scalar path
            // Apply reversed filter
            for (int k = 0; k < filter_len; ++k) {
                int input_idx = k_out + k; // Assuming input is padded correctly for 'full' conv start index
                // Need symmetric padding handling here based on original input_len
                int padded_idx = input_idx; // Placeholder: This needs the actual padding logic
                // Example simple boundary:
                if (padded_idx >=0 && padded_idx < input_len_padded){ // Use padded length for buffer access
                     // Read filter reversed
                    conv_val += input[padded_idx] * filter[filter_len - 1 - k];
                }
            }
        }
        // Write output (potentially vectorized write if output_len is aligned)
        output[k_out] = conv_val;
    }
}


//----------------------------------------------------------------------------
// Device Helper: Downsample (Vectorized)
//----------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE, bool USE_VEC = (std::is_same<scalar_t, float>::value)>
__device__ void downsample2_vec(
    const scalar_t* input, int input_len,
    scalar_t* output, int output_len,
    int phase = 0)
{
    int tid = threadIdx.x;
    const bool can_vectorize = USE_VEC && (output_len % (VEC_SIZE / 2) == 0) && (phase == 0); // Vectorize pairs

    if (can_vectorize) {
         // Read float4 (covers 2 output elements), write float2 (or handle carefully)
         // This vectorization is tricky due to the strided read (every other element)
         // Let's stick to scalar for simplicity and robustness here.
         // Vectorization benefit might be minimal for simple strided copy.
         for (int k_out = tid; k_out < output_len; k_out += BLOCK_SIZE) {
            int input_idx = 2 * k_out + phase;
            if (input_idx < input_len) {
                output[k_out] = input[input_idx];
            }
         }

    } else { // Scalar path
        for (int k_out = tid; k_out < output_len; k_out += BLOCK_SIZE) {
            int input_idx = 2 * k_out + phase;
            if (input_idx < input_len) {
                output[k_out] = input[input_idx];
            }
        }
    }
}


//----------------------------------------------------------------------------
// Device-Side Kernel Implementation (Vectorized Loads/Compute Attempt)
//----------------------------------------------------------------------------
template <typename scalar_t, int BLOCK_SIZE>
__global__ void wavelet_dwt_multi_level_kernel(
    const scalar_t* __restrict__ input_ptr,
    const scalar_t* __restrict__ filters_ptr,
          scalar_t* __restrict__ output_ptr,
    int N, int D_in, int NumFilterPairs, int FilterLength, int MaxLevels, int OutputDim,
    int input_stride_N, int filters_stride_N, int filters_stride_Pair, int output_stride_N)
{
    // Grid, Block indices
    const int n_global = blockIdx.x; const int initial_pair_idx = blockIdx.y;
    if (n_global >= N || initial_pair_idx >= NumFilterPairs) return;
    const int tid = threadIdx.x; const int block_dim = blockDim.x;

    // Shared Memory Calculation & Pointers (Padded for vectorization/banks)
    const int PADDING_ALIGN_W = WAVELET_SHARED_MEM_PADDING > VEC_SIZE ? WAVELET_SHARED_MEM_PADDING : VEC_SIZE;
    const int D_in_padded = get_padded_dim_wavelet(D_in, PADDING_ALIGN_W);
    // Padded length for convolution input buffer must handle boundaries
    const int conv_input_padded_len = get_padded_dim_wavelet(D_in + FilterLength - 1, PADDING_ALIGN_W); // Max padded length needed
    const int filter_len_aligned = get_padded_dim_wavelet(FilterLength, PADDING_ALIGN_W); // Pad each filter
    const int filter_pair_len_aligned = 2 * filter_len_aligned; // Padded size for pair

    extern __shared__ unsigned char shared_mem_raw[];
    scalar_t* shared_filters = utils::AsDynamicSharedMemory<scalar_t>(shared_mem_raw); // Size filter_pair_len_aligned
    scalar_t* buffer_A = reinterpret_cast<scalar_t*>(shared_filters + filter_pair_len_aligned); // Size conv_input_padded_len
    scalar_t* buffer_B = reinterpret_cast<scalar_t*>(buffer_A + conv_input_padded_len); // Size conv_input_padded_len
    scalar_t* conv_result = reinterpret_cast<scalar_t*>(buffer_B + conv_input_padded_len); // Size conv_input_padded_len

    // Pointers
    const scalar_t* current_input_ptr = input_ptr + n_global * input_stride_N;
    const scalar_t* base_filters_ptr = filters_ptr + n_global * filters_stride_N;
          scalar_t* base_output_ptr = output_ptr + n_global * output_stride_N;

    // --- Load Initial Input Signal with Symmetric Padding ---
    // Needs FilterLength/2 padding on each side roughly for centered conv
    // Load into buffer_A
    const int pad_size = FilterLength / 2; // Approximate padding needed
    for (int i = tid; i < D_in; i += block_dim) {
         buffer_A[pad_size + i] = current_input_ptr[i]; // Load into center
    }
    // Apply padding in shared memory (symmetric reflection)
    for (int i = tid; i < pad_size; i += block_dim) {
        buffer_A[pad_size - 1 - i] = current_input_ptr[i < D_in ? i : D_in - 1 - (i - D_in)]; // Reflect start (approx)
        buffer_A[pad_size + D_in + i] = current_input_ptr[D_in - 1 - (i < D_in ? i : D_in - 1 - (i - D_in)) ]; // Reflect end (approx)
    }
    // Zero out remaining padding area if conv_input_padded_len > D_in + 2*pad_size
     for(int i = tid + D_in + 2*pad_size; i < conv_input_padded_len; i += block_dim) {
         buffer_A[i] = static_cast<scalar_t>(0.0);
     }

    // --- Load Filter Pair ---
    const scalar_t* current_pair_ptr = base_filters_ptr + initial_pair_idx * filters_stride_Pair;
    scalar_t* lo_d = shared_filters;
    scalar_t* hi_d = shared_filters + filter_len_aligned; // Use aligned offset
    // Parallel load (handle alignment)
    for (int i = tid; i < FilterLength; i += block_dim) {
        lo_d[i] = current_pair_ptr[i];
        hi_d[i] = current_pair_ptr[filters_stride_Pair / 2 + i]; // Assuming stride=2*FilterLength
    }
     // Zero out padding in filter arrays
     for(int i = tid + FilterLength; i < filter_len_aligned; i += block_dim) {
         lo_d[i] = static_cast<scalar_t>(0.0);
         hi_d[i] = static_cast<scalar_t>(0.0);
     }
    __syncthreads();

    // --- Multi-Level DWT Loop ---
    scalar_t* current_level_input_padded = buffer_A; // Input buffer has padding
    scalar_t* next_level_output_base = buffer_B; // Output buffer (coefficients only, no padding needed here)
    int current_signal_len = D_in; // Length of signal being processed

    for (int level = 1; level <= MaxLevels; ++level) {
        int conv_out_len = current_signal_len + FilterLength - 1;
        int cA_len = (current_signal_len + 1) / 2;
        int cD_len = current_signal_len / 2;

        // --- Convolve Low Pass ---
        parallel_convolve_1d_vec<scalar_t, BLOCK_SIZE>(
            current_level_input_padded, // Input includes padding
            current_signal_len + 2 * pad_size, // Total length in shared mem buffer (approx)
            filter_len_aligned,        // Padded length for filter access
            lo_d, FilterLength,        // Actual filter length
            conv_result, conv_out_len
        );
        __syncthreads();

        // --- Downsample Low Pass -> cA ---
        // Input is conv_result, output is start of next_level_output_base
        downsample2_vec<scalar_t, BLOCK_SIZE>(conv_result, conv_out_len, next_level_output_base, cA_len, 0);

        // --- Convolve High Pass ---
        parallel_convolve_1d_vec<scalar_t, BLOCK_SIZE>(
            current_level_input_padded, current_signal_len + 2*pad_size, filter_len_aligned,
            hi_d, FilterLength,
            conv_result, conv_out_len
        );
        __syncthreads();

        // --- Downsample High Pass -> cD ---
        // Input is conv_result, output is second half of next_level_output_base
        downsample2_vec<scalar_t, BLOCK_SIZE>(conv_result, conv_out_len, next_level_output_base + cA_len, cD_len, 0);
        __syncthreads();

        // --- Prepare for Next Level ---
        // Input for next level is the Approximation Coefficients (cA)
        // We need to copy cA back into a padded buffer for the next convolution
        scalar_t* temp_swap = buffer_A; // Point to the buffer we just wrote cA/cD into
        buffer_A = buffer_B;            // Next iteration reads from here (contains cA+cD)
        buffer_B = temp_swap;           // Next iteration writes results here

        current_signal_len = cA_len; // Length of signal for next level
        current_level_input_padded = buffer_A; // Start of buffer containing cA+cD
        next_level_output_base = buffer_B;     // Target for next level's results

        // Apply padding to the current_level_input (which now contains cA)
        // This involves copying cA to the center of the *next* input buffer (now buffer_B)
        // and applying boundary reflection.
        if (level < MaxLevels) { // Only pad if not the last level
            for (int i = tid; i < current_signal_len; i += block_dim) {
                 buffer_B[pad_size + i] = current_level_input_padded[i]; // Copy cA to center of B
            }
             // Apply padding in buffer_B
             for (int i = tid; i < pad_size; i += block_dim) {
                 buffer_B[pad_size - 1 - i] = current_level_input_padded[i < current_signal_len ? i : current_signal_len - 1 - (i - current_signal_len)]; // Reflect start
                 buffer_B[pad_size + current_signal_len + i] = current_level_input_padded[current_signal_len - 1 - (i < current_signal_len ? i : current_signal_len - 1 - (i - current_signal_len)) ]; // Reflect end
             }
             // Zero out remaining padding
             for(int i = tid + current_signal_len + 2*pad_size; i < conv_input_padded_len; i += block_dim) {
                 buffer_B[i] = static_cast<scalar_t>(0.0);
             }
             // Prepare pointers for the next iteration
              current_level_input_padded = buffer_B; // Read from B next time
              next_level_output_base = buffer_A;    // Write to A next time
              __syncthreads(); // Ensure padding is complete
        }
    } // End level loop

    // --- Write Final Coefficients ---
    // Final Approximation coefficients (cA) are in `current_level_input_padded` (unpadded section)
    int final_coeffs_len = current_signal_len;
    int output_offset = n_global * output_stride_N + initial_pair_idx * final_coeffs_len;
    scalar_t* final_coeffs_src = current_level_input_padded + pad_size; // Start of actual coefficients

    // Parallel write (Vectorized Attempt)
    bool use_vec4_out = (final_coeffs_len % VEC_SIZE == 0) && std::is_same<scalar_t, float>::value;
    if(use_vec4_out) {
         for (int i = tid; i < final_coeffs_len / VEC_SIZE; i += block_dim) {
             reinterpret_cast<scalar4*>(base_output_ptr + output_offset)[i] = reinterpret_cast<scalar4*>(final_coeffs_src)[i];
         }
    } else {
         for (int i = tid; i < final_coeffs_len; i += block_dim) {
             base_output_ptr[output_offset + i] = final_coeffs_src[i];
         }
    }
}


//----------------------------------------------------------------------------
// Host-Side Function Implementation (Launcher - Updated Shared Mem)
//----------------------------------------------------------------------------
torch::Tensor wavelet_projection_cuda(
    const torch::Tensor& input_signal, const torch::Tensor& filters, int output_dim)
{
    // --- Validation ---
    TORCH_CHECK(input_signal.is_cuda() && filters.is_cuda(), "Inputs CUDA");
    auto input_c = input_signal.contiguous(); auto filters_c = filters.contiguous();
    TORCH_CHECK(input_c.dim()==2 && filters_c.dim()==3, "Dimensions");
    TORCH_CHECK(input_c.size(0) == filters_c.size(0), "N mismatch");
    const auto dtype = input_c.scalar_type(); TORCH_CHECK(dtype == filters_c.scalar_type(), "Dtype mismatch");

    // --- Dimensions ---
    const int N = input_c.size(0); const int D_in = input_c.size(1);
    const int NumFiltersTotal = filters_c.size(1); const int FilterLength = filters_c.size(2);
    TORCH_CHECK(NumFiltersTotal % 2 == 0, "NumFilters must be even");
    const int NumFilterPairs = NumFiltersTotal / 2;

    // --- Determine Max Levels & Expected Output Dim ---
    int current_len = D_in; int max_levels = 0;
    int min_len = FilterLength > 0 ? FilterLength : 1; // Avoid infinite loop if FilterLength is 0
    while (current_len >= min_len && max_levels < MAX_WPD_LEVELS) {
        current_len = (current_len + 1) / 2; max_levels++;
    }
    if (max_levels == 0 && D_in > 0) max_levels = 1; // Ensure at least one level if input exists
    if (max_levels == 0 && D_in == 0) current_len = 0; // Handle edge case of empty input
    const int FinalCoeffsLen = current_len;
    const int ExpectedOutputDim = NumFilterPairs * FinalCoeffsLen;
    TORCH_CHECK(output_dim == ExpectedOutputDim, "OutputDim mismatch: Python expected ", output_dim, ", but CUDA DWT kernel (L=", max_levels, ") expects ", ExpectedOutputDim);

    // --- Prepare Output Tensor ---
    auto output = torch::empty({N, output_dim}, input_c.options());
    if (N == 0 || output_dim == 0) return output; // Handle empty input/output cases

    // --- Kernel Launch Config ---
    const int BLOCK_SIZE = WAVELET_DWT_CUDA_BLOCK_SIZE;
    const dim3 grid_dim(N, NumFilterPairs);
    const dim3 block_dim(BLOCK_SIZE);

    // --- Shared Memory Calculation (Updated) ---
    const int PADDING_ALIGN_W = WAVELET_SHARED_MEM_PADDING > VEC_SIZE ? WAVELET_SHARED_MEM_PADDING : VEC_SIZE;
    const int conv_input_padded_len = get_padded_dim_wavelet(D_in + FilterLength - 1, PADDING_ALIGN_W);
    const int filter_pair_len_aligned = get_padded_dim_wavelet(2 * FilterLength, PADDING_ALIGN_W);
    // Buffer A (padded input) + Buffer B (padded input) + Filters (padded pair) + Conv Result Buffer
    size_t shared_mem_bytes = (2 * conv_input_padded_len + filter_pair_len_aligned + conv_input_padded_len) * torch::elementSize(dtype);
    cudaDeviceProp* deviceProp = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(shared_mem_bytes <= deviceProp->sharedMemPerBlock, "Requested shared memory exceeds limits: ", shared_mem_bytes, " Needed: ", shared_mem_bytes, " > ", deviceProp->sharedMemPerBlock);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- Launch Kernel ---
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "wavelet_dwt_multi_cuda_launcher_v2", [&] {
        int input_stride_N = input_c.stride(0); int filters_stride_N = filters_c.stride(0);
        int filters_stride_Pair = filters_c.stride(1); int output_stride_N = output.stride(0);
        // Filter stride check (should be 2 * FilterLength if pairs are contiguous)
        TORCH_CHECK(filters_stride_Pair == 2 * filters_c.stride(2), "Filter layout assumption mismatch");

        wavelet_dwt_multi_level_kernel<scalar_t, BLOCK_SIZE><<<grid_dim, block_dim, shared_mem_bytes, stream>>>(
            input_c.data_ptr<scalar_t>(), filters_c.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            N, D_in, NumFilterPairs, FilterLength, max_levels, output_dim,
            input_stride_N, filters_stride_N, filters_stride_Pair, output_stride_N
        );
    });

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace wavelets
} // namespace cuda
} // namespace backend
} // namespace ultra_rwka
