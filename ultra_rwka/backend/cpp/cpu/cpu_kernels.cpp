// ultra_rwka/backend/cpp/cpu/cpu_kernels.cpp

#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>     // For potential SIMD types/intrinsics (though often implicit via ATen ops)
#include <ATen/native/CPUBlas.h>  // For explicit BLAS calls if needed

// Include OpenMP header for pragmas if preferred over at::parallel_for sometimes
#ifdef _OPENMP
#include <omp.h>
#endif

// Common header for macros and constants
#include "../../include/common.h"

// Stability constant
constexpr float K_EPSILON = 1e-12f;


/* =========================================================================
 * FA-KLA CPU Implementation (Highly Optimized CPU Attempt)
 * ========================================================================= */

// CPU implementation for Temperature-Modulated Kernelized Linear Attention.
// Optimized for CPU using:
// - Parallelism over batch and heads via at::parallel_for.
// - Direct pointer access and arithmetic within the tightest loop (time dimension).
// - Leveraging ATen/BLAS for matrix/vector operations (gemv, addr/ger, dot).
// - Minimizing temporary allocations within the loop.
// - Requiring contiguous inputs for performance.
torch::Tensor fa_kla_attention_cpu(
    const torch::Tensor& q_mapped,
    const torch::Tensor& k_mapped,
    const torch::Tensor& v_reshaped,
    const torch::Tensor& temp)
{
    // --- Input Validation (Robust) ---
    TORCH_CHECK(q_mapped.dim() == 4, "q_mapped must be 4D (B, T, H, Df)");
    TORCH_CHECK(k_mapped.dim() == 4, "k_mapped must be 4D (B, T, H, Df)");
    TORCH_CHECK(v_reshaped.dim() == 4, "v_reshaped must be 4D (B, T, H, Dv)");
    TORCH_CHECK(temp.dim() == 3, "temp must be 3D (B, T, H)");
    TORCH_CHECK(q_mapped.sizes() == k_mapped.sizes(), "q_mapped and k_mapped shapes must match");
    TORCH_CHECK(q_mapped.size(0) == v_reshaped.size(0) && q_mapped.size(1) == v_reshaped.size(1) && q_mapped.size(2) == v_reshaped.size(2), "B, T, H dimensions must match for q/k/v");
    TORCH_CHECK(q_mapped.size(0) == temp.size(0) && q_mapped.size(1) == temp.size(1) && q_mapped.size(2) == temp.size(2), "B, T, H dimensions must match for q/k and temp");
    TORCH_CHECK(q_mapped.device().is_cpu() && k_mapped.device().is_cpu() && v_reshaped.device().is_cpu() && temp.device().is_cpu(), "All inputs must be on CPU");
    const auto dtype = q_mapped.scalar_type();
    TORCH_CHECK(dtype == k_mapped.scalar_type() && dtype == v_reshaped.scalar_type() && dtype == temp.scalar_type(), "All inputs must have the same dtype");
    TORCH_CHECK(q_mapped.is_contiguous() && k_mapped.is_contiguous() && v_reshaped.is_contiguous() && temp.is_contiguous(),
                "Performance critical: All input tensors must be contiguous for fa_kla_attention_cpu");

    // --- Get Dimensions ---
    const int64_t B = q_mapped.size(0);
    const int64_t T = q_mapped.size(1);
    const int64_t H = q_mapped.size(2);
    const int64_t Df = q_mapped.size(3); // Df_head
    const int64_t Dv = v_reshaped.size(3);

    // --- Output Tensor ---
    // Allocate output tensor (consider memory format for potential performance gains)
    auto output = torch::empty_like(v_reshaped, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    // --- Parallel Computation over Batch and Heads ---
    at::parallel_for(0, B * H, 0, [&](int64_t start, int64_t end) {
        // Per-thread temporary buffer for scaled keys (k_t * temp_bt)
        // Using std::vector<float> if float is the primary type, adjust if using double/half
        std::vector<float> k_t_scaled_buffer(Df); // Allocate once per thread

        for (int64_t bh_idx = start; bh_idx < end; ++bh_idx) {
            const int64_t b = bh_idx / H;
            const int64_t h = bh_idx % H;

            // Per-thread state tensors initialized to zero
            auto kv_state = torch::zeros({Df, Dv}, q_mapped.options());
            auto k_sum_state = torch::zeros({Df}, q_mapped.options());

            // Type dispatching for pointer types and operations
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "fa_kla_attention_cpu_inner_v3", [&] {
                // --- Get Data Pointers (safe due to contiguous check) ---
                const scalar_t* const q_ptr = q_mapped.data_ptr<scalar_t>();
                const scalar_t* const k_ptr = k_mapped.data_ptr<scalar_t>();
                const scalar_t* const v_ptr = v_reshaped.data_ptr<scalar_t>();
                const scalar_t* const t_ptr = temp.data_ptr<scalar_t>();
                      scalar_t* out_ptr = output.data_ptr<scalar_t>();
                      scalar_t* kv_state_ptr = kv_state.data_ptr<scalar_t>();
                      scalar_t* k_sum_state_ptr = k_sum_state.data_ptr<scalar_t>();

                // --- Calculate Strides (for pointer arithmetic) ---
                const int64_t q_stride_T = H * Df; const int64_t q_stride_H = Df;
                const int64_t k_stride_T = H * Df; const int64_t k_stride_H = Df;
                const int64_t v_stride_T = H * Dv; const int64_t v_stride_H = Dv;
                const int64_t t_stride_T = H;
                const int64_t out_stride_T = H * Dv; const int64_t out_stride_H = Dv;

                // --- Base Pointers for this Batch/Head ---
                const scalar_t* base_q_ptr = q_ptr + b * T * q_stride_T + h * q_stride_H;
                const scalar_t* base_k_ptr = k_ptr + b * T * k_stride_T + h * k_stride_H;
                const scalar_t* base_v_ptr = v_ptr + b * T * v_stride_T + h * v_stride_H;
                const scalar_t* base_t_ptr = t_ptr + b * T * t_stride_T + h;
                      scalar_t* base_out_ptr = out_ptr + b * T * out_stride_T + h * out_stride_H;

                // --- Time Step Loop (Sequential Recurrence) ---
                for (int64_t t_step = 0; t_step < T; ++t_step) {
                    // --- Pointers to Current Time Step Data ---
                    const scalar_t* q_t_ptr = base_q_ptr + t_step * q_stride_T;
                    const scalar_t* k_t_ptr = base_k_ptr + t_step * k_stride_T;
                    const scalar_t* v_t_ptr = base_v_ptr + t_step * v_stride_T;
                          scalar_t* out_t_ptr = base_out_ptr + t_step * out_stride_T;
                    const scalar_t temp_bt = base_t_ptr[t_step * t_stride_T];

                    // --- State Update ---
                    // 1. k_t_scaled = k_t * temp_bt (store in buffer)
                    //    Manual loop - compiler likely to vectorize this (SIMD).
                    //    Using vector buffer cast to scalar_t* for alignment potential.
                    scalar_t* k_t_scaled_ptr = reinterpret_cast<scalar_t*>(k_t_scaled_buffer.data());
                    #pragma omp simd // Optional hint for vectorization
                    for (int64_t d = 0; d < Df; ++d) {
                        k_t_scaled_ptr[d] = k_t_ptr[d] * temp_bt;
                    }

                    // 2. k_sum_state += k_t_scaled
                    //    Manual loop - compiler likely to vectorize.
                    #pragma omp simd
                    for (int64_t d = 0; d < Df; ++d) {
                        k_sum_state_ptr[d] += k_t_scaled_ptr[d];
                    }

                    // 3. kv_state += outer(k_t_scaled, v_t) (BLAS Level 2: ger)
                    //    Use ATen's interface which calls optimized BLAS (MKL, OpenBLAS).
                    //    Create temporary views without copying data.
                    auto k_t_scaled_view = torch::from_blob(k_t_scaled_ptr, {Df}, kv_state.options());
                    auto v_t_view = torch::from_blob(const_cast<scalar_t*>(v_t_ptr), {Dv}, kv_state.options());
                    auto kv_state_view = torch::from_blob(kv_state_ptr, {Df, Dv}, kv_state.strides(), kv_state.options());
                    kv_state_view.addr_(k_t_scaled_view, v_t_view, static_cast<scalar_t>(1.0)); // In-place ger

                    // --- Calculate Output ---
                    // 1. Numerator: num_t = q_t @ kv_state (BLAS Level 2: gemv)
                    //    y = alpha*A*x + beta*y => out_t = 1.0 * kv_state^T @ q_t + 0.0 * out_t
                    auto q_t_view = torch::from_blob(const_cast<scalar_t*>(q_t_ptr), {Df}, kv_state.options());
                    auto out_t_view = torch::from_blob(out_t_ptr, {Dv}, kv_state.options());
                    // Note: kv_state is (Df, Dv), kv_state.t() is (Dv, Df)
                    // We need q_t @ kv_state -> (1, Df) @ (Df, Dv) -> (1, Dv)
                    // So, gemv signature: y = alpha*A*x + beta*y
                    // y = out_t_view (Dv), A = kv_state (Df, Dv), x = q_t_view (Df), alpha=1, beta=0
                    // This doesn't match standard gemv A*x. We need x^T * A.
                    // Alternative: matmul(q_t.unsqueeze(0), kv_state)
                    // Let's use matmul for clarity, relying on its optimized backend.
                    auto num_t = at::matmul(q_t_view.unsqueeze(0), kv_state_view); // (1, Df) @ (Df, Dv) -> (1, Dv)

                    // 2. Denominator: den_t = dot(q_t, k_sum_state) (BLAS Level 1: dot)
                    auto k_sum_state_view = torch::from_blob(k_sum_state_ptr, {Df}, kv_state.options());
                    scalar_t den_t = at::dot(q_t_view, k_sum_state_view).item<scalar_t>();

                    // 3. Final output: out_t = num_t / (den_t + epsilon)
                    const scalar_t den_stable_inv = static_cast<scalar_t>(1.0) / (den_t + static_cast<scalar_t>(K_EPSILON));
                    // Copy numerator result to output, then scale in-place
                    out_t_view.copy_(num_t.squeeze(0));
                    out_t_view.mul_(den_stable_inv); // In-place scaling

                } // end time loop
            }); // end AT_DISPATCH
        } // end parallel loop body
    }); // end at::parallel_for

    return output;
}


/* =========================================================================
 * Wavelet Projection CPU Implementation (Clarified Placeholder)
 * ========================================================================= */

// Optimized CPU implementation for the *placeholder* wavelet projection.
// Uses efficient grouped convolution via torch::conv1d.
// Aggregation uses reshape/indexing trick with fallback loop.
// WARNING: This function's logic (Conv1d + Sum) is a placeholder based on the
//          Python code structure. A real wavelet implementation (e.g., DWT)
//          would require a different algorithm (e.g., recursive filtering and
//          downsampling). Optimizing this placeholder further is not meaningful
//          without specifying the exact wavelet transform required by the research.
torch::Tensor wavelet_projection_cpu(
    const torch::Tensor& input,
    const torch::Tensor& filters)
{
    // --- Input Validation ---
    TORCH_CHECK(input.dim() == 2, "input must be 2D (N, D_in)");
    TORCH_CHECK(filters.dim() == 3, "filters must be 3D (N, NumFilters, FilterLength)");
    TORCH_CHECK(input.size(0) == filters.size(0), "Batch dimension N must match");
    TORCH_CHECK(input.is_contiguous() && filters.is_contiguous(), "Input and filters must be contiguous");
    TORCH_CHECK(input.device().is_cpu() && filters.device().is_cpu(), "All inputs must be on CPU");

    // --- Get Dimensions ---
    const int64_t N = input.size(0);
    const int64_t D_in = input.size(1);
    const int64_t NumFilters = filters.size(1);
    const int64_t FilterLength = filters.size(2);
    const int64_t OutputDim = NumFilters; // Placeholder: Output dim assumed equal to NumFilters

    // --- Prepare for Grouped Convolution ---
    auto input_conv = input.unsqueeze(1); // (N, 1, D_in)
    auto filters_conv = filters.view({N * NumFilters, 1, FilterLength}); // (N*NumF, 1, FiltL)

    // --- Perform Grouped Convolution (Optimized Backend) ---
    auto conv_options = torch::nn::Conv1dOptions(1, N * NumFilters, FilterLength)
                            .stride(1).padding(0).groups(N).bias(false);
    auto conv_out = torch::conv1d(input_conv, filters_conv, torch::nullopt,
                                  conv_options.stride(), conv_options.padding(),
                                  conv_options.dilation(), conv_options.groups());
    // Output shape: (N, N*NumFilters, L_out) where L_out = D_in - FilterLength + 1

    // --- Aggregate Placeholder Result (Sum over L_out) ---
    // Attempt optimized reshape/index method
    int64_t L_out = conv_out.size(2);
    if (L_out <= 0) { // Handle cases where filter is longer than input
        return torch::zeros({N, OutputDim}, input.options());
    }
    try {
        // Reshape to expose the per-group filter outputs
        auto reshaped_out = conv_out.view({N, N, NumFilters, L_out});
        // Extract the diagonal corresponding to applying filter bank 'i' to input 'i'
        auto diag_indices = torch::arange(N, torch::kLong);
        auto extracted_diag = reshaped_out.index({diag_indices, diag_indices}); // Shape (N, NumFilters, L_out)
        // Aggregate over the conv output length (L_out)
        auto output = extracted_diag.sum(2); // Shape (N, NumFilters)
        return output;
    } catch (const std::exception& e) {
         // Fallback to parallel loop if view/indexing fails
         TORCH_WARN("Optimized wavelet aggregation via reshape/index failed, falling back to loop. Error: ", e.what());
         auto output = torch::zeros({N, OutputDim}, input.options());
         at::parallel_for(0, N, 0, [&](int64_t start, int64_t end) { // Use at::parallel_for
            for (int64_t i = start; i < end; ++i) {
                 // Slice carefully: batch item i, channels for group i
                 auto item_conv_out = conv_out.slice(0, i, i + 1) // Select batch item i -> (1, N*NumF, L_out)
                                        .slice(1, i * NumFilters, (i + 1) * NumFilters); // Select channels -> (1, NumF, L_out)
                 auto aggregated_i = item_conv_out.sum(2); // Sum over L_out -> (1, NumF)
                 output.slice(0, i, i + 1) = aggregated_i;
            }
         });
         return output;
    }
}


/* =========================================================================
 * iDAM CPU Implementation (Optimized - Relies on ATen)
 * ========================================================================= */

// Optimized iDAM retrieval. Leverages ATen/BLAS for core operations.
// Includes parallel distance calculation for shared keys.
std::tuple<torch::Tensor, torch::Tensor> idam_retrieve_cpu(
    const torch::Tensor& query_keys,
    const torch::Tensor& buffer,
    const torch::Tensor& learnable_keys,
    const torch::Tensor& temperature)
{
    // --- Input Validation ---
    TORCH_CHECK(buffer.dim() == 3, "buffer must be 3D (B, Nb, Dv)");
    const int ndim_query = query_keys.dim();
    TORCH_CHECK(ndim_query == 2 || ndim_query == 3, "query_keys must be 2D or 3D");
    const int ndim_lkeys = learnable_keys.dim();
    TORCH_CHECK(ndim_lkeys == 2 || ndim_lkeys == 3, "learnable_keys must be 2D or 3D");
    TORCH_CHECK(learnable_keys.size(-1) == query_keys.size(-1), "Key dimensions must match");
    TORCH_CHECK(learnable_keys.size(-2) == buffer.size(-2), "NumBins must match");
    TORCH_CHECK(buffer.device().is_cpu() && query_keys.device().is_cpu() &&
                learnable_keys.device().is_cpu() && temperature.device().is_cpu(), "All inputs must be on CPU");
    TORCH_CHECK(buffer.is_contiguous() && query_keys.is_contiguous() && learnable_keys.is_contiguous(),
                "Inputs must be contiguous for iDAM retrieve");

    // --- Prepare Tensors & Dimensions ---
    const int64_t B = buffer.size(0);
    const int64_t NumBins = buffer.size(1);
    const int64_t D_key = query_keys.size(-1);
    const int64_t T = (ndim_query == 3) ? query_keys.size(1) : 1;
    const auto dtype = query_keys.scalar_type();

    // --- Calculate Distances ---
    torch::Tensor sq_dists; // Shape (B, T or 1, NumBins)

    if (ndim_lkeys == 2 && B > 1) { // Shared keys: Optimize distance calc
        sq_dists = torch::empty({B, T, NumBins}, query_keys.options());
        auto lkeys_2d = learnable_keys;
        auto query_keys_flat = query_keys.view({B * T, D_key});

        at::parallel_for(0, B * T, 0, [&](int64_t start, int64_t end) {
            // Per-thread temporary diff buffer
            std::vector<float> diff_buffer(D_key);

            for (int64_t bt_idx = start; bt_idx < end; ++bt_idx) {
                int64_t b_idx = bt_idx / T;
                int64_t t_idx = bt_idx % T;
                // Pointer to current query vector data
                 const float* q_vec_ptr = query_keys_flat.data_ptr<float>() + bt_idx * D_key; // Assuming float for now
                 float* dists_out_ptr = sq_dists.data_ptr<float>() + b_idx * T * NumBins + t_idx * NumBins; // Pointer to output slice

                // Loop over learnable keys (bins)
                for(int64_t n=0; n<NumBins; ++n) {
                    const float* lkey_ptr = lkeys_2d.data_ptr<float>() + n * D_key;
                    float current_sq_dist = 0.0f;
                    // Calculate squared diff and sum manually (potential for SIMD by compiler)
                    float* diff_ptr = diff_buffer.data();
                    #pragma omp simd reduction(+:current_sq_dist) // Hint for vectorization + reduction
                    for(int64_t d=0; d<D_key; ++d) {
                        float diff_val = lkey_ptr[d] - q_vec_ptr[d];
                        current_sq_dist += diff_val * diff_val;
                    }
                     dists_out_ptr[n] = current_sq_dist;
                }
            }
        }); // end parallel_for

        if (ndim_query == 2) { sq_dists = sq_dists.squeeze(1); }

    } else { // Per-batch keys or B=1: Use broadcasting subtraction (simpler, often efficient)
        auto q_keys_b = query_keys.unsqueeze(-2);
        auto lkeys_prep = (ndim_lkeys == 2) ? learnable_keys.unsqueeze(0) : learnable_keys; // Add batch dim if needed
        if (ndim_query == 3) { lkeys_prep = lkeys_prep.unsqueeze(1); } // Add time dim if needed
        sq_dists = (q_keys_b - lkeys_prep).pow_(2).sum(-1);
    }

    // --- Calculate Attention Weights ---
    float tau = temperature.item<float>();
    TORCH_CHECK(tau > 0, "Temperature tau must be positive");
    auto logits = sq_dists.mul_(-1.0f / tau);
    auto attn_weights = torch::softmax(logits, /*dim=*/-1);

    // --- Retrieve Values (Use ATen/BLAS via bmm/einsum) ---
    torch::Tensor retrieved_values;
    auto buf = buffer; // Already checked contiguous
    if (ndim_query == 3) { // Attn: (B, T, Nb), Buffer: (B, Nb, Dv)
        retrieved_values = torch::einsum("btn,bnd->btd", {attn_weights, buf});
    } else { // Attn: (B, Nb), Buffer: (B, Nb, Dv)
        retrieved_values = torch::bmm(attn_weights.unsqueeze(1), buf).squeeze(1);
    }

    return std::make_tuple(retrieved_values.contiguous(), attn_weights.contiguous());
}

// Optimized iDAM update (remains largely the same, relies on parallel batch loop and ATen ops)
void idam_update_cpu(
    torch::Tensor& buffer,
    const torch::Tensor& attn_weights,
    const torch::Tensor& values,
    float learning_rate_buffer,
    torch::Tensor& learnable_keys,
    const torch::Tensor& keys,
    float learning_rate_keys)
{
     // --- Validation & Input Prep (mostly unchanged) ---
    TORCH_CHECK(buffer.dim() == 3, "buffer must be 3D (B, Nb, Dv)");
    TORCH_CHECK(values.size(-1) == buffer.size(-1), "D_val match");
    TORCH_CHECK(attn_weights.size(0) == buffer.size(0) && values.size(0) == buffer.size(0), "B match");
    TORCH_CHECK(attn_weights.size(-1) == buffer.size(1), "Nb match");
    bool has_T_dim = (attn_weights.dim() == 3);
    if (has_T_dim) { TORCH_CHECK(attn_weights.size(1) == values.size(1), "T match"); }
    else { TORCH_CHECK(attn_weights.dim() == 2 && values.dim() == 2, "attn/vals must be 2D"); }
    TORCH_CHECK(buffer.is_contiguous(), "Buffer must be contiguous");
    TORCH_CHECK(buffer.device().is_cpu() && attn_weights.device().is_cpu() && values.device().is_cpu(), "CPU only");

    float lambda_buf = learning_rate_buffer;
    float lambda_keys = learning_rate_keys;
    TORCH_CHECK(lambda_buf >= 0.0 && lambda_buf <= 1.0, "lambda_buf range");

    torch::Tensor attn = has_T_dim ? attn_weights.mean(1) : attn_weights;
    torch::Tensor vals = has_T_dim ? values.mean(1) : values;
    attn = attn.contiguous();
    vals = vals.contiguous();

    torch::Tensor update_keys;
    bool update_lkeys_flag = (lambda_keys > 0.0);
    if (update_lkeys_flag) {
        TORCH_CHECK(keys.defined() && learnable_keys.defined(), "keys/lkeys defined");
        TORCH_CHECK(learnable_keys.is_contiguous(), "lkeys contiguous");
        if (has_T_dim) {
             TORCH_CHECK(keys.dim() == 3 && keys.size(1) == attn_weights.size(1), "Keys T dim mismatch");
             update_keys = keys.mean(1);
         } else {
             TORCH_CHECK(keys.dim() == 2, "Keys must be 2D"); update_keys = keys;
         }
        update_keys = update_keys.contiguous();
        TORCH_CHECK(learnable_keys.size(-1) == update_keys.size(-1), "D_key mismatch");
        TORCH_CHECK(learnable_keys.size(-2) == buffer.size(1), "NumBins mismatch");
    }

    const int64_t B = buffer.size(0);

    // --- Parallel Update over Batch ---
    // Uses efficient ATen element-wise ops with broadcasting internally
    at::parallel_for(0, B, 0, [&](int64_t start, int64_t end) {
        for (int64_t b = start; b < end; ++b) {
            auto buffer_b = buffer[b];
            auto attn_b = attn[b].unsqueeze(-1);
            auto vals_b = vals[b].unsqueeze(0);

            // Buffer Update
            buffer_b.mul_(1.0f - lambda_buf * attn_b);
            buffer_b.addcmul_(attn_b, vals_b, lambda_buf);

            // Key Update (only if 3D / per-batch)
            if (update_lkeys_flag && learnable_keys.dim() == 3) {
                auto update_keys_b = update_keys[b].unsqueeze(0);
                auto learnable_keys_b = learnable_keys[b];
                learnable_keys_b.mul_(1.0f - lambda_keys * attn_b);
                learnable_keys_b.addcmul_(attn_b, update_keys_b, lambda_keys);
            }
        }
    });

    // Handle shared key update (outside parallel loop)
     if (update_lkeys_flag && learnable_keys.dim() == 2) {
          auto avg_attn = attn.mean(0).unsqueeze(-1);
          auto avg_keys = update_keys.mean(0).unsqueeze(0);
          learnable_keys.mul_(1.0f - lambda_keys * avg_attn);
          learnable_keys.addcmul_(avg_attn, avg_keys, lambda_keys);
     }
}
