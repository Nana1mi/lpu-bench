#include "scale.h"
#include "lpu_arch.h"
#include "maps_runtime.h"
#include "scale/barrier"
#include "cooperative_groups.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>
#include <cstring>

#include "../../common/timer.h"
#include "../../common/reference.h"

#define K_TILE 64
#define N_VECS 16       // 256 / 16 = 16
#define MAX_M 8
#define HEAD_DIM 64
#define ATTN_NUM_VECS (HEAD_DIM / 16)  // 4

// -------------------------------------------------------------
// Vector Utility Functions
// -------------------------------------------------------------
__device__ KERNEL_M void mpu_gemm_accumulate_tile(
    v16fp32 *Y_acc,
    const v16fp32 *W_tile,
    const float *X_buf,
    unsigned M,
    unsigned k_tile,
    unsigned n_vecs)
{
    for (unsigned m = 0; m < M; ++m) {
        for (unsigned k = 0; k < k_tile; ++k) {
            float x_val = X_buf[m * k_tile + k];
            v16fp32 x_vec = (v16fp32){x_val, x_val, x_val, x_val, x_val, x_val, x_val, x_val,
                                     x_val, x_val, x_val, x_val, x_val, x_val, x_val, x_val};
            const v16fp32 *w_row = W_tile + k * n_vecs;
            for (unsigned v = 0; v < n_vecs; ++v) {
                Y_acc[m * n_vecs + v] += x_vec * w_row[v];
            }
        }
    }
}

// -------------------------------------------------------------
// GEMM Variant 1: Default Compiler Bank Mapping
// -------------------------------------------------------------
__global__ void gemm_bank_default(
    __DDR float *Y,
    const __DDR float *X,
    const __DDR float *W,
    unsigned M, unsigned K, unsigned N,
    __DDR unsigned long long *dev_cycles_out)
{
    if (threadIdx.x != 0) return;

    __mem0__ v16fp32 W_tile[K_TILE * N_VECS] __attribute__((aligned(64)));
    __mem1__ float X_buf[MAX_M * K_TILE] __attribute__((aligned(64)));
    __mem2__ v16fp32 Y_acc[MAX_M * N_VECS] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    unsigned int n_vecs = N / 16;
    for (unsigned i = 0; i < M * n_vecs; ++i) {
        Y_acc[i] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    unsigned num_k_tiles = K / K_TILE;
    for (unsigned t = 0; t < num_k_tiles; ++t) {
        for (unsigned k = 0; k < K_TILE; ++k) {
            const __DDR float *w_row = W + (t * K_TILE + k) * N;
            memcpy_async(block, (float*)(W_tile + k * n_vecs), w_row,
                         scale::aligned_size_t<4>(N * sizeof(float)), bar);
        }
        for (unsigned m = 0; m < M; ++m) {
            const __DDR float *x_src = X + m * K + t * K_TILE;
            memcpy_async(block, X_buf + m * K_TILE, x_src,
                         scale::aligned_size_t<4>(K_TILE * sizeof(float)), bar);
        }
        bar.arrive_and_wait();

        mpu_gemm_accumulate_tile(Y_acc, W_tile, X_buf, M, K_TILE, n_vecs);
        MPU_SYNC();
    }

    for (unsigned m = 0; m < M; ++m) {
        __DDR float *y_dst = Y + m * N;
        memcpy_async(block, y_dst, (float*)(Y_acc + m * n_vecs),
                     scale::aligned_size_t<4>(N * sizeof(float)), bar);
    }
    bar.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_end = tick();
    *dev_cycles_out = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// GEMM Variant 2: Explicit Bank Allocation (__fix1__, __fix2__, __fix3__)
// -------------------------------------------------------------
__global__ void gemm_bank_explicit(
    __DDR float *Y,
    const __DDR float *X,
    const __DDR float *W,
    unsigned M, unsigned K, unsigned N,
    __DDR unsigned long long *dev_cycles_out)
{
    if (threadIdx.x != 0) return;

    __mem0__ __fix1__ v16fp32 W_tile[K_TILE * N_VECS] __attribute__((aligned(64)));
    __mem1__ __fix2__ float X_buf[MAX_M * K_TILE] __attribute__((aligned(64)));
    __mem2__ __fix3__ v16fp32 Y_acc[MAX_M * N_VECS] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    unsigned int n_vecs = N / 16;
    for (unsigned i = 0; i < M * n_vecs; ++i) {
        Y_acc[i] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    unsigned num_k_tiles = K / K_TILE;
    for (unsigned t = 0; t < num_k_tiles; ++t) {
        for (unsigned k = 0; k < K_TILE; ++k) {
            const __DDR float *w_row = W + (t * K_TILE + k) * N;
            memcpy_async(block, (float*)(W_tile + k * n_vecs), w_row,
                         scale::aligned_size_t<4>(N * sizeof(float)), bar);
        }
        for (unsigned m = 0; m < M; ++m) {
            const __DDR float *x_src = X + m * K + t * K_TILE;
            memcpy_async(block, X_buf + m * K_TILE, x_src,
                         scale::aligned_size_t<4>(K_TILE * sizeof(float)), bar);
        }
        bar.arrive_and_wait();

        mpu_gemm_accumulate_tile(Y_acc, W_tile, X_buf, M, K_TILE, n_vecs);
        MPU_SYNC();
    }

    for (unsigned m = 0; m < M; ++m) {
        __DDR float *y_dst = Y + m * N;
        memcpy_async(block, y_dst, (float*)(Y_acc + m * n_vecs),
                     scale::aligned_size_t<4>(N * sizeof(float)), bar);
    }
    bar.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_end = tick();
    *dev_cycles_out = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// GEMM Variant 3: Intentional Bank Conflict (__fix1__ for BOTH W and X)
// -------------------------------------------------------------
__global__ void gemm_bank_conflict(
    __DDR float *Y,
    const __DDR float *X,
    const __DDR float *W,
    unsigned M, unsigned K, unsigned N,
    __DDR unsigned long long *dev_cycles_out)
{
    if (threadIdx.x != 0) return;

    // Intentionally placing both W_tile (64KB) and X_buf (2KB) into __fix1__
    __mem0__ __fix1__ v16fp32 W_tile[K_TILE * N_VECS] __attribute__((aligned(64)));
    __mem1__ __fix1__ float X_buf[MAX_M * K_TILE] __attribute__((aligned(64)));
    __mem2__ __fix2__ v16fp32 Y_acc[MAX_M * N_VECS] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    unsigned int n_vecs = N / 16;
    for (unsigned i = 0; i < M * n_vecs; ++i) {
        Y_acc[i] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    unsigned num_k_tiles = K / K_TILE;
    for (unsigned t = 0; t < num_k_tiles; ++t) {
        for (unsigned k = 0; k < K_TILE; ++k) {
            const __DDR float *w_row = W + (t * K_TILE + k) * N;
            memcpy_async(block, (float*)(W_tile + k * n_vecs), w_row,
                         scale::aligned_size_t<4>(N * sizeof(float)), bar);
        }
        for (unsigned m = 0; m < M; ++m) {
            const __DDR float *x_src = X + m * K + t * K_TILE;
            memcpy_async(block, X_buf + m * K_TILE, x_src,
                         scale::aligned_size_t<4>(K_TILE * sizeof(float)), bar);
        }
        bar.arrive_and_wait();

        mpu_gemm_accumulate_tile(Y_acc, W_tile, X_buf, M, K_TILE, n_vecs);
        MPU_SYNC();
    }

    for (unsigned m = 0; m < M; ++m) {
        __DDR float *y_dst = Y + m * N;
        memcpy_async(block, y_dst, (float*)(Y_acc + m * n_vecs),
                     scale::aligned_size_t<4>(N * sizeof(float)), bar);
    }
    bar.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_end = tick();
    *dev_cycles_out = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// Attention Helpers
// -------------------------------------------------------------
__device__ inline float fast_expf(float x) {
    if (x < -88.0f) return 0.0f;
    return std::exp(x);
}

__device__ void fast_softmax(float *scores, float *probs, unsigned L) {
    float max_s = -1e30f;
    for (unsigned i = 0; i < L; ++i) {
        if (scores[i] > max_s) max_s = scores[i];
    }
    float sum_exp = 0.0f;
    for (unsigned i = 0; i < L; ++i) {
        float p = fast_expf(scores[i] - max_s);
        probs[i] = p;
        sum_exp += p;
    }
    float inv_sum = 1.0f / (sum_exp > 1e-12f ? sum_exp : 1.0f);
    for (unsigned i = 0; i < L; ++i) {
        probs[i] *= inv_sum;
    }
}

__device__ KERNEL_M void mpu_attn_scores_tile(
    const v16fp32 *q_vecs,
    const v16fp32 *K_tile,
    float *scores_out,
    unsigned T,
    float scale)
{
    for (unsigned i = 0; i < T; ++i) {
        const v16fp32 *k_ptr = K_tile + i * ATTN_NUM_VECS;
        v16fp32 p0 = q_vecs[0] * k_ptr[0];
        v16fp32 p1 = q_vecs[1] * k_ptr[1];
        v16fp32 p2 = q_vecs[2] * k_ptr[2];
        v16fp32 p3 = q_vecs[3] * k_ptr[3];
        v16fp32 p_sum = p0 + p1 + p2 + p3;
        const float *p_f = (const float*)&p_sum;
        float dot = 0.0f;
        for (int j = 0; j < 16; ++j) {
            dot += p_f[j];
        }
        scores_out[i] = dot * scale;
    }
}

__device__ KERNEL_M void mpu_attn_accumulate_v_tile(
    v16fp32 *out_vecs,
    const v16fp32 *V_tile,
    const float *probs,
    unsigned T)
{
    for (unsigned i = 0; i < T; ++i) {
        float p_val = probs[i];
        v16fp32 p_vec = (v16fp32){p_val, p_val, p_val, p_val, p_val, p_val, p_val, p_val,
                                 p_val, p_val, p_val, p_val, p_val, p_val, p_val, p_val};
        const v16fp32 *v_ptr = V_tile + i * ATTN_NUM_VECS;
        out_vecs[0] += p_vec * v_ptr[0];
        out_vecs[1] += p_vec * v_ptr[1];
        out_vecs[2] += p_vec * v_ptr[2];
        out_vecs[3] += p_vec * v_ptr[3];
    }
}

// -------------------------------------------------------------
// Attention Variant A: Compact Region (RESIDENT_L = 128, 32 KB)
// -------------------------------------------------------------
__global__ void attn_compact_128(
    __DDR float *O,
    const __DDR float *Q,
    const __DDR float *K,
    const __DDR float *V,
    unsigned actual_L,
    unsigned d,
    float scale,
    __DDR unsigned long long *dev_cycles_out)
{
    if (threadIdx.x != 0) return;

    // Compact buffer: 128 * 4 * 16 * 4 = 32 KB per bank
    __mem0__ v16fp32 K_resident[128 * ATTN_NUM_VECS] __attribute__((aligned(64)));
    __mem1__ v16fp32 V_resident[128 * ATTN_NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 Q_buf[ATTN_NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 O_acc[ATTN_NUM_VECS] __attribute__((aligned(64)));
    __mem3__ float scores[128] __attribute__((aligned(64)));
    __mem4__ float probs[128] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

    // Preload KV Cache into DM
    memcpy_async(block, K_resident, K, scale::aligned_size_t<4>(actual_L * d * sizeof(float)), bar);
    memcpy_async(block, V_resident, V, scale::aligned_size_t<4>(actual_L * d * sizeof(float)), bar);
    bar.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    memcpy_async(block, Q_buf, Q, scale::aligned_size_t<4>(d * sizeof(float)), bar);
    bar.arrive_and_wait();

    for (unsigned v = 0; v < ATTN_NUM_VECS; ++v) {
        O_acc[v] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    mpu_attn_scores_tile(Q_buf, K_resident, scores, actual_L, scale);
    MPU_SYNC();

    fast_softmax(scores, probs, actual_L);

    mpu_attn_accumulate_v_tile(O_acc, V_resident, probs, actual_L);
    MPU_SYNC();

    memcpy_async(block, O, (float*)O_acc, scale::aligned_size_t<4>(d * sizeof(float)), bar);
    bar.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_end = tick();
    *dev_cycles_out = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// Attention Variant B: Large Region (RESIDENT_L = 1024, 256 KB)
// Computing the same actual_L = 128
// -------------------------------------------------------------
__global__ void attn_large_region_1024(
    __DDR float *O,
    const __DDR float *Q,
    const __DDR float *K,
    const __DDR float *V,
    unsigned actual_L,
    unsigned d,
    float scale,
    __DDR unsigned long long *dev_cycles_out)
{
    if (threadIdx.x != 0) return;

    // Large buffer: 1024 * 4 * 16 * 4 = 256 KB per bank
    __mem0__ v16fp32 K_resident[1024 * ATTN_NUM_VECS] __attribute__((aligned(64)));
    __mem1__ v16fp32 V_resident[1024 * ATTN_NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 Q_buf[ATTN_NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 O_acc[ATTN_NUM_VECS] __attribute__((aligned(64)));
    __mem3__ float scores[1024] __attribute__((aligned(64)));
    __mem4__ float probs[1024] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

    // Preload KV Cache into DM (only load actual_L tokens)
    memcpy_async(block, K_resident, K, scale::aligned_size_t<4>(actual_L * d * sizeof(float)), bar);
    memcpy_async(block, V_resident, V, scale::aligned_size_t<4>(actual_L * d * sizeof(float)), bar);
    bar.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    memcpy_async(block, Q_buf, Q, scale::aligned_size_t<4>(d * sizeof(float)), bar);
    bar.arrive_and_wait();

    for (unsigned v = 0; v < ATTN_NUM_VECS; ++v) {
        O_acc[v] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    mpu_attn_scores_tile(Q_buf, K_resident, scores, actual_L, scale);
    MPU_SYNC();

    fast_softmax(scores, probs, actual_L);

    mpu_attn_accumulate_v_tile(O_acc, V_resident, probs, actual_L);
    MPU_SYNC();

    memcpy_async(block, O, (float*)O_acc, scale::aligned_size_t<4>(d * sizeof(float)), bar);
    bar.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_end = tick();
    *dev_cycles_out = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// Host Dispatcher & Main
// -------------------------------------------------------------
int main(int argc, char **argv) {
    std::string test = "gemm_default";
    unsigned M = 1;
    unsigned K = 256;
    unsigned N = 256;
    unsigned actual_L = 128;
    unsigned d = 64;
    int repeats = 3;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test" && i + 1 < argc) test = argv[++i];
        else if (arg == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc) actual_L = std::atoi(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
    }

    mapsStream_t stream;
    mapsStreamCreate(&stream);
    HostEventTimer timer(stream);
    dim3 blockDim(1, 1, 1);
    dim3 gridDim(1, 1, 1);

    unsigned long long *d_cycles = nullptr;
    uint64_t d_cycles_pa = 0;
    mapsMalloc((void**)&d_cycles, sizeof(unsigned long long));
    mapsMemobjGetPhysicalAddress(d_cycles, &d_cycles_pa);

    if (test == "gemm_default" || test == "gemm_explicit" || test == "gemm_conflict") {
        size_t x_size = M * K * sizeof(float);
        size_t w_size = K * N * sizeof(float);
        size_t y_size = M * N * sizeof(float);

        std::vector<float> h_X(M * K);
        std::vector<float> h_W(K * N);
        std::vector<float> h_Y(M * N, 0.0f);
        std::vector<float> h_Y_ref(M * N, 0.0f);

        for (size_t i = 0; i < M * K; ++i) h_X[i] = static_cast<float>((int)(i % 13) - 6) * 0.1f;
        for (size_t i = 0; i < K * N; ++i) h_W[i] = static_cast<float>((int)(i % 17) - 8) * 0.05f;
        lpu_bench::gemm_cpu_ref(h_X.data(), h_W.data(), h_Y_ref.data(), M, K, N);

        float *d_X = nullptr, *d_W = nullptr, *d_Y = nullptr;
        uint64_t d_X_pa = 0, d_W_pa = 0, d_Y_pa = 0;
        mapsMalloc((void**)&d_X, x_size);
        mapsMalloc((void**)&d_W, w_size);
        mapsMalloc((void**)&d_Y, y_size);
        mapsMemobjGetPhysicalAddress(d_X, &d_X_pa);
        mapsMemobjGetPhysicalAddress(d_W, &d_W_pa);
        mapsMemobjGetPhysicalAddress(d_Y, &d_Y_pa);

        mapsMemcpy(d_X, h_X.data(), x_size, mapsMemcpyHostToDevice);
        mapsMemcpy(d_W, h_W.data(), w_size, mapsMemcpyHostToDevice);

        std::cout << "--- CSV DATA START ---" << std::endl;
        std::cout << "test,config,M,K,N,repeat,sim_ms,sim_cycles,device_cycles,wall_ms,correct,max_diff" << std::endl;

        for (int r = 0; r < repeats; ++r) {
            mapsMemset(d_Y, 0, y_size);
            unsigned long long init_c = 0;
            mapsMemcpy(d_cycles, &init_c, sizeof(unsigned long long), mapsMemcpyHostToDevice);

            timer.start();
            if (test == "gemm_default") {
                gemm_bank_default<<<gridDim, blockDim, 0, stream>>>(
                    (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
                    M, K, N, (__DDR unsigned long long*)d_cycles_pa);
            } else if (test == "gemm_explicit") {
                gemm_bank_explicit<<<gridDim, blockDim, 0, stream>>>(
                    (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
                    M, K, N, (__DDR unsigned long long*)d_cycles_pa);
            } else if (test == "gemm_conflict") {
                gemm_bank_conflict<<<gridDim, blockDim, 0, stream>>>(
                    (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
                    M, K, N, (__DDR unsigned long long*)d_cycles_pa);
            }
            timer.stop();

            unsigned long long h_cycles = 0;
            mapsMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), mapsMemcpyDeviceToHost);
            mapsMemcpy(h_Y.data(), d_Y, y_size, mapsMemcpyDeviceToHost);

            float max_diff = 0.0f;
            bool pass = lpu_bench::verify_results(h_Y_ref.data(), h_Y.data(), M * N, 1e-2f, 1e-3f, &max_diff);
            TimingResult t_res = timer.get_result(h_cycles);

            std::cout << test << "," << test << "," << M << "," << K << "," << N << ","
                      << r << ","
                      << t_res.sim_ms << ","
                      << static_cast<unsigned long long>(t_res.sim_ms) << ","
                      << h_cycles << ","
                      << t_res.wall_clock_ms << ","
                      << (pass ? "true" : "false") << ","
                      << max_diff << std::endl;
        }
        std::cout << "--- CSV DATA END ---" << std::endl;

        mapsFree(d_X);
        mapsFree(d_W);
        mapsFree(d_Y);
    } else if (test == "attn_compact" || test == "attn_large") {
        size_t q_size = d * sizeof(float);
        size_t kv_size = actual_L * d * sizeof(float);
        size_t o_size = d * sizeof(float);
        float scale = 1.0f / std::sqrt(static_cast<float>(d));

        std::vector<float> h_Q(d);
        std::vector<float> h_K(actual_L * d);
        std::vector<float> h_V(actual_L * d);
        std::vector<float> h_O(d, 0.0f);
        std::vector<float> h_O_ref(d, 0.0f);

        for (size_t i = 0; i < d; ++i) h_Q[i] = static_cast<float>((int)(i % 7) - 3) * 0.1f;
        for (size_t i = 0; i < actual_L * d; ++i) {
            h_K[i] = static_cast<float>((int)(i % 11) - 5) * 0.05f;
            h_V[i] = static_cast<float>((int)(i % 13) - 6) * 0.05f;
        }

        lpu_bench::decode_attn_cpu_ref(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), actual_L, d);

        float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
        uint64_t d_Q_pa = 0, d_K_pa = 0, d_V_pa = 0, d_O_pa = 0;
        mapsMalloc((void**)&d_Q, q_size);
        mapsMalloc((void**)&d_K, kv_size);
        mapsMalloc((void**)&d_V, kv_size);
        mapsMalloc((void**)&d_O, o_size);
        mapsMemobjGetPhysicalAddress(d_Q, &d_Q_pa);
        mapsMemobjGetPhysicalAddress(d_K, &d_K_pa);
        mapsMemobjGetPhysicalAddress(d_V, &d_V_pa);
        mapsMemobjGetPhysicalAddress(d_O, &d_O_pa);

        mapsMemcpy(d_Q, h_Q.data(), q_size, mapsMemcpyHostToDevice);
        mapsMemcpy(d_K, h_K.data(), kv_size, mapsMemcpyHostToDevice);
        mapsMemcpy(d_V, h_V.data(), kv_size, mapsMemcpyHostToDevice);

        std::cout << "--- CSV DATA START ---" << std::endl;
        std::cout << "test,config,M,K,N,repeat,sim_ms,sim_cycles,device_cycles,wall_ms,correct,max_diff" << std::endl;

        for (int r = 0; r < repeats; ++r) {
            mapsMemset(d_O, 0, o_size);
            unsigned long long init_c = 0;
            mapsMemcpy(d_cycles, &init_c, sizeof(unsigned long long), mapsMemcpyHostToDevice);

            timer.start();
            if (test == "attn_compact") {
                attn_compact_128<<<gridDim, blockDim, 0, stream>>>(
                    (__DDR float*)d_O_pa, (const __DDR float*)d_Q_pa,
                    (const __DDR float*)d_K_pa, (const __DDR float*)d_V_pa,
                    actual_L, d, scale, (__DDR unsigned long long*)d_cycles_pa);
            } else if (test == "attn_large") {
                attn_large_region_1024<<<gridDim, blockDim, 0, stream>>>(
                    (__DDR float*)d_O_pa, (const __DDR float*)d_Q_pa,
                    (const __DDR float*)d_K_pa, (const __DDR float*)d_V_pa,
                    actual_L, d, scale, (__DDR unsigned long long*)d_cycles_pa);
            }
            timer.stop();

            unsigned long long h_cycles = 0;
            mapsMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), mapsMemcpyDeviceToHost);
            mapsMemcpy(h_O.data(), d_O, o_size, mapsMemcpyDeviceToHost);

            float max_diff = 0.0f;
            bool pass = lpu_bench::verify_results(h_O_ref.data(), h_O.data(), d, 1e-2f, 1e-3f, &max_diff);
            TimingResult t_res = timer.get_result(h_cycles);

            std::cout << test << "," << (test == "attn_compact" ? "compact_32KB" : "large_256KB") << ",1,"
                      << actual_L << "," << d << ","
                      << r << ","
                      << t_res.sim_ms << ","
                      << static_cast<unsigned long long>(t_res.sim_ms) << ","
                      << h_cycles << ","
                      << t_res.wall_clock_ms << ","
                      << (pass ? "true" : "false") << ","
                      << max_diff << std::endl;
        }
        std::cout << "--- CSV DATA END ---" << std::endl;

        mapsFree(d_Q);
        mapsFree(d_K);
        mapsFree(d_V);
        mapsFree(d_O);
    }

    mapsFree(d_cycles);
    return 0;
}
