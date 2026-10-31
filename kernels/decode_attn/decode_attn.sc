#include "scale.h"
#include "maps_runtime_api.h"
#include "aurora.h"
#include "scale/barrier"
#include "cooperative_groups.h"

#if defined(__aurora_s__) || defined(__confidence__)
#include "llvm/tick.h"
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>

#include "timer.h"
#include "reference.h"

#define D_MODEL   64
#define NUM_VECS  4   // 4 * 16 = 64 floats
#define TILE_L    16  // tokens per tile
#define MAX_L     512 // maximum sequence length supported in DM scratchpad

// -------------------------------------------------------------
// SPU Inline Fast Range-Reduced Exponential & Softmax
// -------------------------------------------------------------
__device__ inline float fast_expf(float x) {
    if (x < -15.0f) return 0.0f;
    // Range reduction: x = k * ln(2) + r, where k = round(x * log2(e))
    float x_log2e = x * 1.4426950408889634f;
    int k = static_cast<int>(x_log2e < 0.0f ? (x_log2e - 0.5f) : (x_log2e + 0.5f));
    float r = x - static_cast<float>(k) * 0.6931471805599453f;

    // Horner evaluation of exp(r) on [-0.35, +0.35]
    float exp_r = 1.0f + r * (1.0f + r * 0.5f * (1.0f + r * 0.33333333f * (1.0f + r * 0.25f)));

    // 2^k via IEEE-754 bit manipulation
    union { int i; float f; } u;
    u.i = (k + 127) << 23;
    float p2k = u.f;

    return exp_r * p2k;
}

__device__ inline void fast_softmax(float *scores, float *probs, unsigned L) {
    float max_s = scores[0];
    for (unsigned i = 1; i < L; ++i) {
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

// -------------------------------------------------------------
// MPU Vector Kernels
// -------------------------------------------------------------
__device__ KERNEL_M void mpu_attn_scores_tile(
    const v16fp32 *q_vecs,
    const v16fp32 *K_tile,
    float *scores_out,
    unsigned T,
    float scale)
{
    for (unsigned i = 0; i < T; ++i) {
        const v16fp32 *k_ptr = K_tile + i * NUM_VECS;
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
        const v16fp32 *v_ptr = V_tile + i * NUM_VECS;
        out_vecs[0] += p_vec * v_ptr[0];
        out_vecs[1] += p_vec * v_ptr[1];
        out_vecs[2] += p_vec * v_ptr[2];
        out_vecs[3] += p_vec * v_ptr[3];
    }
}

// -------------------------------------------------------------
// SPU Kernel v0: DDR Streaming (Serial)
// -------------------------------------------------------------
__global__ void decode_attn_v0_ddr_kernel(
    __DDR float *O,
    const __DDR float *Q,
    const __DDR float *K,
    const __DDR float *V,
    unsigned L,
    unsigned d,
    float scale,
    __DDR unsigned long long *dev_cycles_out)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    __mem0__ v16fp32 K_tile[TILE_L * NUM_VECS] __attribute__((aligned(64)));
    __mem1__ v16fp32 V_tile[TILE_L * NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 Q_buf[NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 O_acc[NUM_VECS] __attribute__((aligned(64)));
    __mem3__ float scores[MAX_L] __attribute__((aligned(64)));
    __mem4__ float probs[MAX_L] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    // 1. Fetch Q (64 floats = 256 bytes)
    memcpy_async(block, Q_buf, Q, scale::aligned_size_t<4>(d * sizeof(float)), bar);
    bar.arrive_and_wait();

    // 2. Initialize O accumulator
    for (unsigned v = 0; v < NUM_VECS; ++v) {
        O_acc[v] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    unsigned num_tiles = L / TILE_L;

    // 3. Pass 1: Compute attention scores (Q * K^T)
    for (unsigned t = 0; t < num_tiles; ++t) {
        const __DDR float *k_src = K + t * TILE_L * d;
        memcpy_async(block, K_tile, k_src, scale::aligned_size_t<4>(TILE_L * d * sizeof(float)), bar);
        bar.arrive_and_wait();

        mpu_attn_scores_tile(Q_buf, K_tile, scores + t * TILE_L, TILE_L, scale);
        MPU_SYNC();
    }

    // 4. Pass 2: Softmax on SPU
    fast_softmax(scores, probs, L);

    // 5. Pass 3: Value accumulation (P * V)
    for (unsigned t = 0; t < num_tiles; ++t) {
        const __DDR float *v_src = V + t * TILE_L * d;
        memcpy_async(block, V_tile, v_src, scale::aligned_size_t<4>(TILE_L * d * sizeof(float)), bar);
        bar.arrive_and_wait();

        mpu_attn_accumulate_v_tile(O_acc, V_tile, probs + t * TILE_L, TILE_L);
        MPU_SYNC();
    }

    // 6. Pass 4: Write back output O
    memcpy_async(block, O, (float*)O_acc, scale::aligned_size_t<4>(d * sizeof(float)), bar);
    bar.arrive_and_wait();

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t_end = tick();
    dev_cycles_out[0] = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// SPU Kernel v1: DM Resident (KV Cache pre-loaded in DM)
// -------------------------------------------------------------
__global__ void decode_attn_v1_dm_resident_kernel(
    __DDR float *O,
    const __DDR float *Q,
    const __DDR float *K,
    const __DDR float *V,
    unsigned L,
    unsigned d,
    float scale,
    __DDR unsigned long long *dev_cycles_out)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    // Up to L=128 tokens resident in DM banks
    __mem0__ v16fp32 K_resident[128 * NUM_VECS] __attribute__((aligned(64)));
    __mem1__ v16fp32 V_resident[128 * NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 Q_buf[NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 O_acc[NUM_VECS] __attribute__((aligned(64)));
    __mem3__ float scores[MAX_L] __attribute__((aligned(64)));
    __mem4__ float probs[MAX_L] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

    // Preload KV Cache into DM (outside timing to measure resident performance)
    memcpy_async(block, K_resident, K, scale::aligned_size_t<4>(L * d * sizeof(float)), bar);
    memcpy_async(block, V_resident, V, scale::aligned_size_t<4>(L * d * sizeof(float)), bar);
    bar.arrive_and_wait();

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    // 1. Fetch Q (256 bytes)
    memcpy_async(block, Q_buf, Q, scale::aligned_size_t<4>(d * sizeof(float)), bar);
    bar.arrive_and_wait();

    // 2. Initialize O accumulator
    for (unsigned v = 0; v < NUM_VECS; ++v) {
        O_acc[v] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    // 3. Compute scores directly from resident DM (zero DDR reads)
    mpu_attn_scores_tile(Q_buf, K_resident, scores, L, scale);
    MPU_SYNC();

    // 4. Softmax
    fast_softmax(scores, probs, L);

    // 5. Accumulate V directly from resident DM (zero DDR reads)
    mpu_attn_accumulate_v_tile(O_acc, V_resident, probs, L);
    MPU_SYNC();

    // 6. Write back output O
    memcpy_async(block, O, (float*)O_acc, scale::aligned_size_t<4>(d * sizeof(float)), bar);
    bar.arrive_and_wait();

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t_end = tick();
    dev_cycles_out[0] = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// SPU Kernel v2: DDR Overlap (Double Buffering DMA + MPU)
// -------------------------------------------------------------
__global__ void decode_attn_v2_overlap_kernel(
    __DDR float *O,
    const __DDR float *Q,
    const __DDR float *K,
    const __DDR float *V,
    unsigned L,
    unsigned d,
    float scale,
    __DDR unsigned long long *dev_cycles_out)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    __mem0__ v16fp32 KV_buf0[TILE_L * NUM_VECS] __attribute__((aligned(64)));
    __mem1__ v16fp32 KV_buf1[TILE_L * NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 Q_buf[NUM_VECS] __attribute__((aligned(64)));
    __mem2__ v16fp32 O_acc[NUM_VECS] __attribute__((aligned(64)));
    __mem3__ float scores[MAX_L] __attribute__((aligned(64)));
    __mem4__ float probs[MAX_L] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar0, bar1;
    if (threadIdx.x == 0) {
        init(&bar0, 1);
        init(&bar1, 1);
    }
    __syncthreads();

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    // 1. Fetch Q
    memcpy_async(block, Q_buf, Q, scale::aligned_size_t<4>(d * sizeof(float)), bar0);
    bar0.arrive_and_wait();

    for (unsigned v = 0; v < NUM_VECS; ++v) {
        O_acc[v] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    unsigned num_tiles = L / TILE_L;

    // 2. Pass 1: Double-buffered Q * K^T
    // Prologue: prefetch tile 0 into KV_buf0
    memcpy_async(block, KV_buf0, K, scale::aligned_size_t<4>(TILE_L * d * sizeof(float)), bar0);
    bar0.arrive_and_wait();

    for (unsigned t = 0; t < num_tiles - 1; ++t) {
        if (t % 2 == 0) {
            memcpy_async(block, KV_buf1, K + (t + 1) * TILE_L * d, scale::aligned_size_t<4>(TILE_L * d * sizeof(float)), bar1);
            mpu_attn_scores_tile(Q_buf, KV_buf0, scores + t * TILE_L, TILE_L, scale);
            MPU_SYNC();
            bar1.arrive_and_wait();
        } else {
            memcpy_async(block, KV_buf0, K + (t + 1) * TILE_L * d, scale::aligned_size_t<4>(TILE_L * d * sizeof(float)), bar0);
            mpu_attn_scores_tile(Q_buf, KV_buf1, scores + t * TILE_L, TILE_L, scale);
            MPU_SYNC();
            bar0.arrive_and_wait();
        }
    }
    // Epilogue for K
    if ((num_tiles - 1) % 2 == 0) {
        mpu_attn_scores_tile(Q_buf, KV_buf0, scores + (num_tiles - 1) * TILE_L, TILE_L, scale);
    } else {
        mpu_attn_scores_tile(Q_buf, KV_buf1, scores + (num_tiles - 1) * TILE_L, TILE_L, scale);
    }
    MPU_SYNC();

    // 3. Pass 2: Softmax
    fast_softmax(scores, probs, L);

    // 4. Pass 3: Double-buffered P * V
    // Prologue: prefetch tile 0 into KV_buf0
    memcpy_async(block, KV_buf0, V, scale::aligned_size_t<4>(TILE_L * d * sizeof(float)), bar0);
    bar0.arrive_and_wait();

    for (unsigned t = 0; t < num_tiles - 1; ++t) {
        if (t % 2 == 0) {
            memcpy_async(block, KV_buf1, V + (t + 1) * TILE_L * d, scale::aligned_size_t<4>(TILE_L * d * sizeof(float)), bar1);
            mpu_attn_accumulate_v_tile(O_acc, KV_buf0, probs + t * TILE_L, TILE_L);
            MPU_SYNC();
            bar1.arrive_and_wait();
        } else {
            memcpy_async(block, KV_buf0, V + (t + 1) * TILE_L * d, scale::aligned_size_t<4>(TILE_L * d * sizeof(float)), bar0);
            mpu_attn_accumulate_v_tile(O_acc, KV_buf1, probs + t * TILE_L, TILE_L);
            MPU_SYNC();
            bar0.arrive_and_wait();
        }
    }
    // Epilogue for V
    if ((num_tiles - 1) % 2 == 0) {
        mpu_attn_accumulate_v_tile(O_acc, KV_buf0, probs + (num_tiles - 1) * TILE_L, TILE_L);
    } else {
        mpu_attn_accumulate_v_tile(O_acc, KV_buf1, probs + (num_tiles - 1) * TILE_L, TILE_L);
    }
    MPU_SYNC();

    // 5. Pass 4: Write back output O
    memcpy_async(block, O, (float*)O_acc, scale::aligned_size_t<4>(d * sizeof(float)), bar0);
    bar0.arrive_and_wait();

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t_end = tick();
    dev_cycles_out[0] = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// Host Main Program
// -------------------------------------------------------------
int main(int argc, char** argv) {
    std::string version = "v0";
    unsigned L = 128;
    unsigned d = 64;
    int repeats = 3;

    if (argc >= 2) version = argv[1];
    if (argc >= 3) L = std::atoi(argv[2]);
    if (argc >= 4) d = std::atoi(argv[3]);
    if (argc >= 5) repeats = std::atoi(argv[4]);

    if (d != D_MODEL) {
        std::cerr << "Error: d (" << d << ") must be " << D_MODEL << std::endl;
        return 1;
    }
    if (L > MAX_L) {
        std::cerr << "Error: L (" << L << ") exceeds MAX_L (" << MAX_L << ")" << std::endl;
        return 1;
    }
    if (version == "v1" && L > 128) {
        std::cerr << "Error: v1_dm_resident requires L <= 128 (hardware DM bank limit)" << std::endl;
        return 1;
    }

    std::cout << "[ATTN] Running configuration: Version=" << version
              << ", L=" << L << ", d=" << d
              << ", Repeats=" << repeats << std::endl;

    size_t q_size = d * sizeof(float);
    size_t kv_size = L * d * sizeof(float);
    size_t o_size = d * sizeof(float);

    float *h_Q = nullptr, *h_K = nullptr, *h_V = nullptr, *h_O = nullptr, *h_O_ref = nullptr;
    mapsHostAlloc((void**)&h_Q, q_size, 0);
    mapsHostAlloc((void**)&h_K, kv_size, 0);
    mapsHostAlloc((void**)&h_V, kv_size, 0);
    mapsHostAlloc((void**)&h_O, o_size, 0);
    h_O_ref = new float[d];

    // Initialize deterministically
    for (size_t j = 0; j < d; ++j) {
        int v = static_cast<int>(j % 7) - 3;
        h_Q[j] = static_cast<float>(v) * 0.1f;
    }
    for (size_t i = 0; i < L * d; ++i) {
        int vk = static_cast<int>(i % 11) - 5;
        h_K[i] = static_cast<float>(vk) * 0.05f;
        int vv = static_cast<int>((i * 2) % 13) - 6;
        h_V[i] = static_cast<float>(vv) * 0.05f;
    }

    // CPU Reference
    lpu_bench::decode_attn_cpu_ref(h_Q, h_K, h_V, h_O_ref, L, d);

    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_O = nullptr;
    unsigned long long *d_cycles = nullptr;
    unsigned long long *h_cycles = nullptr;
    mapsHostAlloc((void**)&h_cycles, sizeof(unsigned long long), 0);

    mapsMalloc((void**)&d_Q, q_size);
    mapsMalloc((void**)&d_K, kv_size);
    mapsMalloc((void**)&d_V, kv_size);
    mapsMalloc((void**)&d_O, o_size);
    mapsMalloc((void**)&d_cycles, sizeof(unsigned long long));

    uint64_t d_Q_pa = 0, d_K_pa = 0, d_V_pa = 0, d_O_pa = 0, d_cycles_pa = 0;
    mapsMemobjGetPhysicalAddress(d_Q, &d_Q_pa);
    mapsMemobjGetPhysicalAddress(d_K, &d_K_pa);
    mapsMemobjGetPhysicalAddress(d_V, &d_V_pa);
    mapsMemobjGetPhysicalAddress(d_O, &d_O_pa);
    mapsMemobjGetPhysicalAddress(d_cycles, &d_cycles_pa);

    mapsMemcpy(d_Q, h_Q, q_size, mapsMemcpyHostToDevice);
    mapsMemcpy(d_K, h_K, kv_size, mapsMemcpyHostToDevice);
    mapsMemcpy(d_V, h_V, kv_size, mapsMemcpyHostToDevice);

    mapsStream_t stream;
    mapsStreamCreate(&stream);
    HostEventTimer timer(stream);

    dim3 blockDim(1, 1, 1);
    dim3 gridDim(1, 1, 1);
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d));

    // Warm-up & Correctness Verification Run
    if (version == "v0") {
        decode_attn_v0_ddr_kernel<<<gridDim, blockDim, 0, stream>>>(
            (__DDR float*)d_O_pa, (const __DDR float*)d_Q_pa,
            (const __DDR float*)d_K_pa, (const __DDR float*)d_V_pa,
            L, d, scale_factor, (__DDR unsigned long long*)d_cycles_pa);
    } else if (version == "v1") {
        decode_attn_v1_dm_resident_kernel<<<gridDim, blockDim, 0, stream>>>(
            (__DDR float*)d_O_pa, (const __DDR float*)d_Q_pa,
            (const __DDR float*)d_K_pa, (const __DDR float*)d_V_pa,
            L, d, scale_factor, (__DDR unsigned long long*)d_cycles_pa);
    } else {
        decode_attn_v2_overlap_kernel<<<gridDim, blockDim, 0, stream>>>(
            (__DDR float*)d_O_pa, (const __DDR float*)d_Q_pa,
            (const __DDR float*)d_K_pa, (const __DDR float*)d_V_pa,
            L, d, scale_factor, (__DDR unsigned long long*)d_cycles_pa);
    }
    mapsStreamSynchronize(stream);
    mapsMemcpy(h_O, d_O, o_size, mapsMemcpyDeviceToHost);

    float max_diff = 0.0f;
    bool pass = lpu_bench::verify_results(h_O_ref, h_O, d, 2e-2f, 2e-2f, &max_diff);
    if (!pass) {
        std::cerr << "[ERROR] Correctness verification FAILED for " << version << "!" << std::endl;
        return 2;
    }

    double flops = 4.0 * static_cast<double>(L) * static_cast<double>(d) + 3.0 * static_cast<double>(L);
    double total_bytes = (version == "v1") ? (2.0 * d * sizeof(float)) : (2.0 * L * d * sizeof(float) + 2.0 * d * sizeof(float));

    std::cout << "--- CSV DATA START ---" << std::endl;
    std::cout << "kernel,version,L,d,dtype,repeat,sim_ms,sim_cycles,device_cycles,wall_ms,total_bytes,flops,effective_bw_gbps,correct" << std::endl;

    for (int r = 0; r < repeats; ++r) {
        h_cycles[0] = 0;
        mapsMemcpyAsync(d_cycles, h_cycles, sizeof(unsigned long long), mapsMemcpyHostToDevice, stream);

        timer.start();
        if (version == "v0") {
            decode_attn_v0_ddr_kernel<<<gridDim, blockDim, 0, stream>>>(
                (__DDR float*)d_O_pa, (const __DDR float*)d_Q_pa,
                (const __DDR float*)d_K_pa, (const __DDR float*)d_V_pa,
                L, d, scale_factor, (__DDR unsigned long long*)d_cycles_pa);
        } else if (version == "v1") {
            decode_attn_v1_dm_resident_kernel<<<gridDim, blockDim, 0, stream>>>(
                (__DDR float*)d_O_pa, (const __DDR float*)d_Q_pa,
                (const __DDR float*)d_K_pa, (const __DDR float*)d_V_pa,
                L, d, scale_factor, (__DDR unsigned long long*)d_cycles_pa);
        } else {
            decode_attn_v2_overlap_kernel<<<gridDim, blockDim, 0, stream>>>(
                (__DDR float*)d_O_pa, (const __DDR float*)d_Q_pa,
                (const __DDR float*)d_K_pa, (const __DDR float*)d_V_pa,
                L, d, scale_factor, (__DDR unsigned long long*)d_cycles_pa);
        }
        timer.stop();

        mapsMemcpyAsync(h_cycles, d_cycles, sizeof(unsigned long long), mapsMemcpyDeviceToHost, stream);
        mapsStreamSynchronize(stream);

        TimingResult res = timer.get_result(h_cycles[0]);
        // Effective BW in GB/s assuming 1.0 GHz device clock (bytes / cycles = GB/s)
        double bw_gbps = (res.device_cycles > 0) ? (total_bytes / static_cast<double>(res.device_cycles)) : 0.0;

        std::cout << "decode_attn,"
                  << version << ","
                  << L << ","
                  << d << ","
                  << "fp32,"
                  << r << ","
                  << std::fixed << std::setprecision(4) << res.sim_ms << ","
                  << res.sim_cycles << ","
                  << res.device_cycles << ","
                  << std::fixed << std::setprecision(2) << res.wall_clock_ms << ","
                  << static_cast<uint64_t>(total_bytes) << ","
                  << static_cast<uint64_t>(flops) << ","
                  << std::fixed << std::setprecision(4) << bw_gbps << ","
                  << "PASS" << std::endl;
    }
    std::cout << "--- CSV DATA END ---" << std::endl;

    mapsStreamDestroy(stream);
    mapsFree(d_Q);
    mapsFree(d_K);
    mapsFree(d_V);
    mapsFree(d_O);
    mapsFree(d_cycles);
    mapsFreeHost(h_Q);
    mapsFreeHost(h_K);
    mapsFreeHost(h_V);
    mapsFreeHost(h_O);
    mapsFreeHost(h_cycles);
    delete[] h_O_ref;

    return 0;
}
