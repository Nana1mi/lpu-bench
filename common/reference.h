#ifndef LPU_BENCH_REFERENCE_H
#define LPU_BENCH_REFERENCE_H

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace lpu_bench {

// -------------------------------------------------------------
// CPU Reference GEMM: Y = X * W
// X: [M, K], W: [K, N], Y: [M, N]
// -------------------------------------------------------------
inline void gemm_cpu_ref(const float *X, const float *W, float *Y, int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += (double)X[m * K + k] * (double)W[k * N + n];
            }
            Y[m * N + n] = static_cast<float>(sum);
        }
    }
}

// -------------------------------------------------------------
// CPU Reference Decode Attention:
// q: [1, d], K_cache: [L, d], V_cache: [L, d], Output: [1, d]
// O = softmax(q * K^T / sqrt(d)) * V
// -------------------------------------------------------------
inline void decode_attn_cpu_ref(const float *q, const float *K_cache, const float *V_cache,
                                float *out, int L, int d) {
    std::vector<double> scores(L, 0.0);
    double scale = 1.0 / std::sqrt(static_cast<double>(d));

    // 1. S = (q * K^T) / sqrt(d)
    double max_score = -1e30;
    for (int i = 0; i < L; ++i) {
        double dot = 0.0;
        for (int j = 0; j < d; ++j) {
            dot += static_cast<double>(q[j]) * static_cast<double>(K_cache[i * d + j]);
        }
        scores[i] = dot * scale;
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    // 2. Softmax: P = exp(S - max_score) / sum(exp(S - max_score))
    double sum_exp = 0.0;
    std::vector<double> probs(L, 0.0);
    for (int i = 0; i < L; ++i) {
        probs[i] = std::exp(scores[i] - max_score);
        sum_exp += probs[i];
    }
    double inv_sum = 1.0 / (sum_exp > 1e-12 ? sum_exp : 1.0);
    for (int i = 0; i < L; ++i) {
        probs[i] *= inv_sum;
    }

    // 3. O = P * V
    for (int j = 0; j < d; ++j) {
        double weighted_val = 0.0;
        for (int i = 0; i < L; ++i) {
            weighted_val += probs[i] * static_cast<double>(V_cache[i * d + j]);
        }
        out[j] = static_cast<float>(weighted_val);
    }
}

// -------------------------------------------------------------
// Precision Verification Helper (Absolute & Relative Tolerance)
// -------------------------------------------------------------
inline bool verify_results(const float *ref, const float *test, int total_elements,
                           float atol = 1e-2f, float rtol = 1e-3f, float *out_max_diff = nullptr) {
    float max_diff = 0.0f;
    int mismatch_count = 0;

    for (int i = 0; i < total_elements; ++i) {
        float diff = std::fabs(ref[i] - test[i]);
        float rel_diff = diff / (std::fabs(ref[i]) + 1e-6f);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > atol && rel_diff > rtol) {
            if (mismatch_count < 5) {
                std::cerr << "[VERIFY FAIL] Index " << i << ": ref=" << ref[i]
                          << ", test=" << test[i] << ", diff=" << diff
                          << ", rel_diff=" << rel_diff << std::endl;
            }
            mismatch_count++;
        }
    }

    if (out_max_diff) {
        *out_max_diff = max_diff;
    }

    if (mismatch_count > 0) {
        std::cerr << "[VERIFY] Total mismatches: " << mismatch_count << " / " << total_elements
                  << " (max diff: " << max_diff << ")" << std::endl;
        return false;
    }

    std::cout << "[VERIFY] PASS! Checked " << total_elements << " elements (max diff: " << max_diff << ")" << std::endl;
    return true;
}

} // namespace lpu_bench

#endif // LPU_BENCH_REFERENCE_H
