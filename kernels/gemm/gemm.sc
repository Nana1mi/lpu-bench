#include "scale.h"
#include "maps_runtime_api.h"
#include "lpu_arch.h"
#include "scale/barrier"
#include "cooperative_groups.h"

#if LPU_IS_SPU || defined(__confidence__)
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

#define K_TILE 16
#define N_VECS 16  // 16 * 16 = 256 floats per row
#define MAX_M  32

// -------------------------------------------------------------
// MPU Vector Kernel (C++ SIMD)
// -------------------------------------------------------------
__device__ KERNEL_M void mpu_gemm_accumulate_tile(
    v16fp32 *Y_acc,          // [M, N_VECS]
    const v16fp32 *W_tile,   // [K_TILE, N_VECS]
    const float *X_tile,     // [M, K_TILE]
    unsigned M,
    unsigned k_steps,
    unsigned n_vecs)
{
    for (unsigned m = 0; m < M; ++m) {
        for (unsigned k = 0; k < k_steps; ++k) {
            float x_val = X_tile[m * k_steps + k];
            v16fp32 x_vec = (v16fp32){x_val, x_val, x_val, x_val, x_val, x_val, x_val, x_val,
                                     x_val, x_val, x_val, x_val, x_val, x_val, x_val, x_val};
            for (unsigned v = 0; v < n_vecs; ++v) {
                Y_acc[m * n_vecs + v] += x_vec * W_tile[k * n_vecs + v];
            }
        }
    }
}

// -------------------------------------------------------------
// SPU Kernel v0: Serial (Fetch -> Compute -> Fetch -> Compute)
// -------------------------------------------------------------
__global__ void gemm_v0_serial_kernel(
    __DDR float *Y,
    const __DDR float *X,
    const __DDR float *W,
    unsigned M,
    unsigned K,
    unsigned N,
    __DDR unsigned long long *dev_cycles_out)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

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

    // 1. Initialize Y accumulator
    for (unsigned i = 0; i < M * N_VECS; ++i) {
        Y_acc[i] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    unsigned num_k_tiles = K / K_TILE;
    for (unsigned t = 0; t < num_k_tiles; ++t) {
        // Fetch W tile [K_TILE, N]
        const __DDR float *w_src = W + t * K_TILE * N;
        memcpy_async(block, W_tile, w_src, scale::aligned_size_t<4>(K_TILE * N * sizeof(float)), bar);

        // Fetch X tile [M, K_TILE]
        for (unsigned m = 0; m < M; ++m) {
            const __DDR float *x_src = X + m * K + t * K_TILE;
            memcpy_async(block, X_buf + m * K_TILE, x_src, scale::aligned_size_t<4>(K_TILE * sizeof(float)), bar);
        }
        bar.arrive_and_wait();

        // MPU compute
        mpu_gemm_accumulate_tile(Y_acc, W_tile, X_buf, M, K_TILE, N_VECS);
        MPU_SYNC();
    }

    // Write back Y [M, N]
    for (unsigned m = 0; m < M; ++m) {
        __DDR float *y_dst = Y + m * N;
        memcpy_async(block, y_dst, (float*)(Y_acc + m * N_VECS), scale::aligned_size_t<4>(N * sizeof(float)), bar);
    }
    bar.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_end = tick();
    dev_cycles_out[0] = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// SPU Kernel v1: Overlap (Double Buffering DMA + MPU async_enqueue)
// -------------------------------------------------------------
__global__ void gemm_v1_overlap_kernel(
    __DDR float *Y,
    const __DDR float *X,
    const __DDR float *W,
    unsigned M,
    unsigned K,
    unsigned N,
    __DDR unsigned long long *dev_cycles_out)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    __mem0__ v16fp32 W_buf0[K_TILE * N_VECS] __attribute__((aligned(64)));
    __mem1__ v16fp32 W_buf1[K_TILE * N_VECS] __attribute__((aligned(64)));
    __mem2__ float X_buf0[MAX_M * K_TILE] __attribute__((aligned(64)));
    __mem3__ float X_buf1[MAX_M * K_TILE] __attribute__((aligned(64)));
    __mem4__ v16fp32 Y_acc[MAX_M * N_VECS] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar0, bar1;
    if (threadIdx.x == 0) {
        init(&bar0, 1);
        init(&bar1, 1);
    }
    __syncthreads();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    // 1. Initialize Y accumulator
    for (unsigned i = 0; i < M * N_VECS; ++i) {
        Y_acc[i] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    unsigned num_k_tiles = K / K_TILE;

    // Prologue: Prefetch tile 0 into W_buf0 and X_buf0
    memcpy_async(block, W_buf0, W, scale::aligned_size_t<4>(K_TILE * N * sizeof(float)), bar0);
    for (unsigned m = 0; m < M; ++m) {
        memcpy_async(block, X_buf0 + m * K_TILE, X + m * K, scale::aligned_size_t<4>(K_TILE * sizeof(float)), bar0);
    }
    bar0.arrive_and_wait();

    // Pipelined main loop: double-buffered W and X, async_enqueue for MPU
    for (unsigned t = 0; t < num_k_tiles - 1; ++t) {
        if (t % 2 == 0) {
            // Compute on W_buf0 & X_buf0 while prefetching tile t+1 into W_buf1 & X_buf1
            memcpy_async(block, W_buf1, W + (t + 1) * K_TILE * N, scale::aligned_size_t<4>(K_TILE * N * sizeof(float)), bar1);
            for (unsigned m = 0; m < M; ++m) {
                memcpy_async(block, X_buf1 + m * K_TILE, X + m * K + (t + 1) * K_TILE, scale::aligned_size_t<4>(K_TILE * sizeof(float)), bar1);
            }
            async_enqueue mpu_gemm_accumulate_tile(Y_acc, W_buf0, X_buf0, M, K_TILE, N_VECS);
            while (lpu_s_callm_counter(warpEntryIdx) > 0);
            bar1.arrive_and_wait();
        } else {
            // Compute on W_buf1 & X_buf1 while prefetching tile t+1 into W_buf0 & X_buf0
            memcpy_async(block, W_buf0, W + (t + 1) * K_TILE * N, scale::aligned_size_t<4>(K_TILE * N * sizeof(float)), bar0);
            for (unsigned m = 0; m < M; ++m) {
                memcpy_async(block, X_buf0 + m * K_TILE, X + m * K + (t + 1) * K_TILE, scale::aligned_size_t<4>(K_TILE * sizeof(float)), bar0);
            }
            async_enqueue mpu_gemm_accumulate_tile(Y_acc, W_buf1, X_buf1, M, K_TILE, N_VECS);
            while (lpu_s_callm_counter(warpEntryIdx) > 0);
            bar0.arrive_and_wait();
        }
    }

    // Epilogue: compute final tile
    if ((num_k_tiles - 1) % 2 == 0) {
        async_enqueue mpu_gemm_accumulate_tile(Y_acc, W_buf0, X_buf0, M, K_TILE, N_VECS);
    } else {
        async_enqueue mpu_gemm_accumulate_tile(Y_acc, W_buf1, X_buf1, M, K_TILE, N_VECS);
    }
    while (lpu_s_callm_counter(warpEntryIdx) > 0);

    // Write back Y [M, N]
    for (unsigned m = 0; m < M; ++m) {
        __DDR float *y_dst = Y + m * N;
        memcpy_async(block, y_dst, (float*)(Y_acc + m * N_VECS), scale::aligned_size_t<4>(N * sizeof(float)), bar0);
    }
    bar0.arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_end = tick();
    dev_cycles_out[0] = t_end - t_start;
#endif
}

// -------------------------------------------------------------
// Host Main Program
// -------------------------------------------------------------
int main(int argc, char** argv) {
    std::string version = "v0";
    unsigned M = 8;
    unsigned K = 256;
    unsigned N = 256;
    int repeats = 3;

    if (argc >= 2) version = argv[1];
    if (argc >= 3) M = std::atoi(argv[2]);
    if (argc >= 4) K = std::atoi(argv[3]);
    if (argc >= 5) N = std::atoi(argv[4]);
    if (argc >= 6) repeats = std::atoi(argv[5]);

    if (M > MAX_M) {
        std::cerr << "Error: M (" << M << ") exceeds MAX_M (" << MAX_M << ")" << std::endl;
        return 1;
    }
    if (N != 256) {
        std::cerr << "Error: N (" << N << ") must be 256 (N_VECS=" << N_VECS << " requires N=256)" << std::endl;
        return 1;
    }

    std::cout << "[GEMM] Running configuration: Version=" << version
              << ", M=" << M << ", K=" << K << ", N=" << N
              << ", Repeats=" << repeats << std::endl;

    size_t x_size = M * K * sizeof(float);
    size_t w_size = K * N * sizeof(float);
    size_t y_size = M * N * sizeof(float);

    float *h_X = nullptr, *h_W = nullptr, *h_Y = nullptr, *h_Y_ref = nullptr;
    mapsHostAlloc((void**)&h_X, x_size, 0);
    mapsHostAlloc((void**)&h_W, w_size, 0);
    mapsHostAlloc((void**)&h_Y, y_size, 0);
    h_Y_ref = new float[M * N];

    // Initialize inputs deterministically with small bounded floats
    for (size_t i = 0; i < M * K; ++i) {
        int v = static_cast<int>(i % 13) - 6;
        h_X[i] = static_cast<float>(v) * 0.1f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        int v = static_cast<int>(i % 17) - 8;
        h_W[i] = static_cast<float>(v) * 0.05f;
    }

    // CPU Reference computation
    lpu_bench::gemm_cpu_ref(h_X, h_W, h_Y_ref, M, K, N);

    float *d_X = nullptr, *d_W = nullptr, *d_Y = nullptr;
    unsigned long long *d_cycles = nullptr;
    unsigned long long *h_cycles = nullptr;
    mapsHostAlloc((void**)&h_cycles, sizeof(unsigned long long), 0);

    mapsMalloc((void**)&d_X, x_size);
    mapsMalloc((void**)&d_W, w_size);
    mapsMalloc((void**)&d_Y, y_size);
    mapsMalloc((void**)&d_cycles, sizeof(unsigned long long));

    uint64_t d_X_pa = 0, d_W_pa = 0, d_Y_pa = 0, d_cycles_pa = 0;
    mapsMemobjGetPhysicalAddress(d_X, &d_X_pa);
    mapsMemobjGetPhysicalAddress(d_W, &d_W_pa);
    mapsMemobjGetPhysicalAddress(d_Y, &d_Y_pa);
    mapsMemobjGetPhysicalAddress(d_cycles, &d_cycles_pa);

    mapsMemcpy(d_X, h_X, x_size, mapsMemcpyHostToDevice);
    mapsMemcpy(d_W, h_W, w_size, mapsMemcpyHostToDevice);

    mapsStream_t stream;
    mapsStreamCreate(&stream);
    HostEventTimer timer(stream);

    dim3 blockDim(1, 1, 1);
    dim3 gridDim(1, 1, 1);

    // Warm-up and correctness verification run
    if (version == "v0") {
        gemm_v0_serial_kernel<<<gridDim, blockDim, 0, stream>>>(
            (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
            M, K, N, (__DDR unsigned long long*)d_cycles_pa);
    } else {
        gemm_v1_overlap_kernel<<<gridDim, blockDim, 0, stream>>>(
            (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
            M, K, N, (__DDR unsigned long long*)d_cycles_pa);
    }
    mapsStreamSynchronize(stream);
    mapsMemcpy(h_Y, d_Y, y_size, mapsMemcpyDeviceToHost);

    float max_diff = 0.0f;
    bool pass = lpu_bench::verify_results(h_Y_ref, h_Y, M * N, 1e-2f, 1e-3f, &max_diff);
    if (!pass) {
        std::cerr << "[ERROR] Warm-up correctness verification FAILED for " << version << "!" << std::endl;
        return 2;
    }

    double total_flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);

    std::cout << "--- CSV DATA START ---" << std::endl;
    std::cout << "kernel,version,M,K,N,dtype,repeat,sim_ms,sim_cycles,device_cycles,wall_ms,flops,flops_per_cycle,correct,max_diff" << std::endl;

    for (int r = 0; r < repeats; ++r) {
        h_cycles[0] = 0;
        mapsMemcpyAsync(d_cycles, h_cycles, sizeof(unsigned long long), mapsMemcpyHostToDevice, stream);

        timer.start();
        if (version == "v0") {
            gemm_v0_serial_kernel<<<gridDim, blockDim, 0, stream>>>(
                (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
                M, K, N, (__DDR unsigned long long*)d_cycles_pa);
        } else {
            gemm_v1_overlap_kernel<<<gridDim, blockDim, 0, stream>>>(
                (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
                M, K, N, (__DDR unsigned long long*)d_cycles_pa);
        }
        timer.stop();

        mapsMemcpyAsync(h_cycles, d_cycles, sizeof(unsigned long long), mapsMemcpyDeviceToHost, stream);
        mapsStreamSynchronize(stream);

        TimingResult res = timer.get_result(h_cycles[0]);
        double flops_per_cycle = (res.device_cycles > 0) ? (total_flops / static_cast<double>(res.device_cycles)) : 0.0;

        // T11: verify results on every repeat outside timing interval
        mapsMemcpy(h_Y, d_Y, y_size, mapsMemcpyDeviceToHost);
        float r_max_diff = 0.0f;
        bool r_pass = lpu_bench::verify_results(h_Y_ref, h_Y, M * N, 1e-2f, 1e-3f, &r_max_diff);
        std::string status = r_pass ? "PASS" : "FAIL";

        std::cout << "gemm,"
                  << version << ","
                  << M << ","
                  << K << ","
                  << N << ","
                  << "fp32,"
                  << r << ","
                  << std::fixed << std::setprecision(4) << res.sim_ms << ","
                  << res.sim_cycles << ","
                  << res.device_cycles << ","
                  << std::fixed << std::setprecision(2) << res.wall_clock_ms << ","
                  << static_cast<uint64_t>(total_flops) << ","
                  << std::fixed << std::setprecision(4) << flops_per_cycle << ","
                  << status << ","
                  << std::scientific << std::setprecision(6) << r_max_diff << std::endl;
    }
    std::cout << "--- CSV DATA END ---" << std::endl;

    mapsStreamDestroy(stream);
    mapsFree(d_X);
    mapsFree(d_W);
    mapsFree(d_Y);
    mapsFree(d_cycles);
    mapsFreeHost(h_X);
    mapsFreeHost(h_W);
    mapsFreeHost(h_Y);
    mapsFreeHost(h_cycles);
    delete[] h_Y_ref;

    return 0;
}
