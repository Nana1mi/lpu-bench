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
#include <algorithm>

#include "timer.h"
#include "reference.h"

#define K_TILE 16
#define N_VECS 16  // 16 * 16 = 256 floats
#define MAX_M  32

// -------------------------------------------------------------
// MPU Vector Kernel (C++ SIMD)
// -------------------------------------------------------------
__device__ KERNEL_M void mpu_gemm_accumulate_tile(
    v16fp32 *Y_acc,          // [M, n_vecs]
    const v16fp32 *W_tile,   // [K_TILE, n_vecs]
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
// SPU Multi-Core Kernel: supports num_threads = 1, 2, 4 (runtime blockDim.x)
// -------------------------------------------------------------
__global__ void gemm_multicore_kernel(
    __DDR float *Y,
    const __DDR float *X,
    const __DDR float *W,
    unsigned M,
    unsigned K,
    unsigned N,
    __DDR unsigned long long *dev_cycles_out)
{
    unsigned int tid = threadIdx.x;
    unsigned int num_threads = blockDim.x;

    if (tid >= num_threads) return;

    // Local memory per core (allocated in core-private local memory)
    __mem0__ v16fp32 W_tile[K_TILE * N_VECS] __attribute__((aligned(64)));
    __mem1__ float X_buf[MAX_M * K_TILE] __attribute__((aligned(64)));
    __mem2__ v16fp32 Y_acc[MAX_M * N_VECS] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar[4];

    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            init(&bar[i], 1);
        }
    }
    __syncthreads();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_start = tick();
#endif

    unsigned int n_cols = N / num_threads;
    unsigned int n_vecs = n_cols / 16;
    unsigned int col_start = tid * n_cols;

    // 1. Initialize Y accumulator
    for (unsigned i = 0; i < M * n_vecs; ++i) {
        Y_acc[i] = (v16fp32){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    unsigned num_k_tiles = K / K_TILE;
    for (unsigned t = 0; t < num_k_tiles; ++t) {
        // Fetch W tile slice for this thread: K_TILE rows, each n_cols floats
        for (unsigned k = 0; k < K_TILE; ++k) {
            const __DDR float *w_row = W + (t * K_TILE + k) * N + col_start;
            memcpy_async(block, (float*)(W_tile + k * n_vecs), w_row,
                         scale::aligned_size_t<4>(n_cols * sizeof(float)), bar[tid]);
        }

        // Fetch X tile [M, K_TILE]
        for (unsigned m = 0; m < M; ++m) {
            const __DDR float *x_src = X + m * K + t * K_TILE;
            memcpy_async(block, X_buf + m * K_TILE, x_src,
                         scale::aligned_size_t<4>(K_TILE * sizeof(float)), bar[tid]);
        }

        bar[tid].arrive_and_wait();

        // MPU compute
        mpu_gemm_accumulate_tile(Y_acc, W_tile, X_buf, M, K_TILE, n_vecs);
        MPU_SYNC();
    }

    // Write back Y slice: M rows, each n_cols floats
    for (unsigned m = 0; m < M; ++m) {
        __DDR float *y_dst = Y + m * N + col_start;
        memcpy_async(block, y_dst, (float*)(Y_acc + m * n_vecs),
                     scale::aligned_size_t<4>(n_cols * sizeof(float)), bar[tid]);
    }
    bar[tid].arrive_and_wait();

#if LPU_IS_SPU || defined(__confidence__)
    unsigned long long t_end = tick();
    dev_cycles_out[tid] = t_end - t_start;
#endif

    __syncthreads();
}

// -------------------------------------------------------------
// Host Main Program
// -------------------------------------------------------------
int main(int argc, char **argv) {
    unsigned M = 1;
    unsigned K = 256;
    unsigned N = 256;
    int repeats = 3;
    unsigned num_threads = 1;
    std::string mode = "N";  // "N" or "C"

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--repeats" && i + 1 < argc) repeats = std::atoi(argv[++i]);
        else if (arg == "--threads" && i + 1 < argc) num_threads = std::atoi(argv[++i]);
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
    }

    if (num_threads != 1 && num_threads != 2 && num_threads != 4) {
        std::cerr << "Error: num_threads must be 1, 2, or 4!" << std::endl;
        return 1;
    }
    if (mode == "C" && num_threads != 4) {
        std::cerr << "Error: C mode currently tested with 4 threads!" << std::endl;
        return 1;
    }

    size_t x_size = M * K * sizeof(float);
    size_t w_size = K * N * sizeof(float);
    size_t y_size = M * N * sizeof(float);

    std::vector<float> h_X(M * K);
    std::vector<float> h_W(K * N);
    std::vector<float> h_Y(M * N, 0.0f);
    std::vector<float> h_Y_ref(M * N, 0.0f);

    for (size_t i = 0; i < M * K; ++i) {
        int v = static_cast<int>(i % 13) - 6;
        h_X[i] = static_cast<float>(v) * 0.1f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        int v = static_cast<int>(i % 17) - 8;
        h_W[i] = static_cast<float>(v) * 0.05f;
    }

    lpu_bench::gemm_cpu_ref(h_X.data(), h_W.data(), h_Y_ref.data(), M, K, N);

    float *d_X = nullptr, *d_W = nullptr, *d_Y = nullptr;
    unsigned long long *d_cycles = nullptr;
    uint64_t d_X_pa = 0, d_W_pa = 0, d_Y_pa = 0, d_cycles_pa = 0;

    mapsMalloc((void**)&d_X, x_size);
    mapsMalloc((void**)&d_W, w_size);
    mapsMalloc((void**)&d_Y, y_size);
    mapsMalloc((void**)&d_cycles, 4 * sizeof(unsigned long long));

    mapsMemobjGetPhysicalAddress(d_X, &d_X_pa);
    mapsMemobjGetPhysicalAddress(d_W, &d_W_pa);
    mapsMemobjGetPhysicalAddress(d_Y, &d_Y_pa);
    mapsMemobjGetPhysicalAddress(d_cycles, &d_cycles_pa);

    mapsMemcpy(d_X, h_X.data(), x_size, mapsMemcpyHostToDevice);
    mapsMemcpy(d_W, h_W.data(), w_size, mapsMemcpyHostToDevice);

    mapsStream_t stream;
    mapsStreamCreate(&stream);
    HostEventTimer timer(stream);

    dim3 blockDim(num_threads, 1, 1);
    dim3 gridDim(1, 1, 1);

    // Warm-up / sanity run with verification
    mapsMemset(d_Y, 0, y_size);
    unsigned long long init_cycles[4] = {0, 0, 0, 0};
    mapsMemcpy(d_cycles, init_cycles, 4 * sizeof(unsigned long long), mapsMemcpyHostToDevice);

    if (mode == "C") {
        mapsLaunchConfig_t config = {0};
        config.blockDim = blockDim;
        config.gridDim = gridDim;
        config.stream = stream;

        mapsLaunchAttribute attr[1];
        attr[0].id = mapsLaunchAttributeConcurrencyThread;
        attr[0].val.concurrencyThread = 1;
        config.attrs = attr;
        config.numAttrs = 1;

        void* args[7] = {
            (void*)&d_Y_pa,
            (void*)&d_X_pa,
            (void*)&d_W_pa,
            (void*)&M,
            (void*)&K,
            (void*)&N,
            (void*)&d_cycles_pa
        };
        mapsLaunchKernelExc(&config, (void*)&gemm_multicore_kernel, args);
    } else {
        gemm_multicore_kernel<<<gridDim, blockDim, 0, stream>>>(
            (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
            M, K, N, (__DDR unsigned long long*)d_cycles_pa);
    }
    mapsStreamSynchronize(stream);
    mapsMemcpy(h_Y.data(), d_Y, y_size, mapsMemcpyDeviceToHost);

    float max_diff = 0.0f;
    bool pass = lpu_bench::verify_results(h_Y_ref.data(), h_Y.data(), M * N, 1e-2f, 1e-3f, &max_diff);
    if (!pass) {
        std::cerr << "[ERROR] Verification FAILED for mode=" << mode << " threads=" << num_threads << "! max_diff=" << max_diff << std::endl;
        return 1;
    }

    double total_flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
    // Total bytes transferred over DDR: W full tile (K*N*4) + num_threads * (M*K*4) for X + full Y writeback (M*N*4)
    double total_bytes = static_cast<double>(K * N * 4 + num_threads * M * K * 4 + M * N * 4);

    std::cout << "--- CSV DATA START ---" << std::endl;
    std::cout << "kernel,version,exec_mode,num_threads,M,K,N,dtype,repeat,sim_ms,sim_cycles,device_cycles,wall_ms,flops,flops_per_cycle,total_bytes,bytes_per_cycle,correct,max_diff" << std::endl;

    for (int r = 0; r < repeats; ++r) {
        mapsMemset(d_Y, 0, y_size);
        mapsMemcpyAsync(d_cycles, init_cycles, 4 * sizeof(unsigned long long), mapsMemcpyHostToDevice, stream);

        timer.start();
        if (mode == "C") {
            mapsLaunchConfig_t config = {0};
            config.blockDim = blockDim;
            config.gridDim = gridDim;
            config.stream = stream;

            mapsLaunchAttribute attr[1];
            attr[0].id = mapsLaunchAttributeConcurrencyThread;
            attr[0].val.concurrencyThread = 1;
            config.attrs = attr;
            config.numAttrs = 1;

            void* args[7] = {
                (void*)&d_Y_pa,
                (void*)&d_X_pa,
                (void*)&d_W_pa,
                (void*)&M,
                (void*)&K,
                (void*)&N,
                (void*)&d_cycles_pa
            };
            mapsLaunchKernelExc(&config, (void*)&gemm_multicore_kernel, args);
        } else {
            gemm_multicore_kernel<<<gridDim, blockDim, 0, stream>>>(
                (__DDR float*)d_Y_pa, (const __DDR float*)d_X_pa, (const __DDR float*)d_W_pa,
                M, K, N, (__DDR unsigned long long*)d_cycles_pa);
        }
        timer.stop();

        unsigned long long h_cycles[4] = {0, 0, 0, 0};
        mapsMemcpyAsync(h_cycles, d_cycles, 4 * sizeof(unsigned long long), mapsMemcpyDeviceToHost, stream);
        mapsStreamSynchronize(stream);

        mapsMemcpy(h_Y.data(), d_Y, y_size, mapsMemcpyDeviceToHost);
        float rep_max_diff = 0.0f;
        bool rep_pass = lpu_bench::verify_results(h_Y_ref.data(), h_Y.data(), M * N, 1e-2f, 1e-3f, &rep_max_diff);

        unsigned long long max_dev_cycles = 0;
        for (unsigned t = 0; t < num_threads; ++t) {
            if (h_cycles[t] > max_dev_cycles) max_dev_cycles = h_cycles[t];
        }

        TimingResult res = timer.get_result(max_dev_cycles);
        double flops_per_cycle = (res.device_cycles > 0) ? (total_flops / static_cast<double>(res.device_cycles)) : 0.0;
        double bytes_per_cycle = (res.device_cycles > 0) ? (total_bytes / static_cast<double>(res.device_cycles)) : 0.0;

        std::cout << "gemm,v0," << mode << "," << num_threads << ","
                  << M << "," << K << "," << N << ",fp32,"
                  << r << ","
                  << std::fixed << std::setprecision(4) << res.sim_ms << ","
                  << res.sim_cycles << ","
                  << res.device_cycles << ","
                  << std::fixed << std::setprecision(2) << res.wall_clock_ms << ","
                  << std::fixed << std::setprecision(0) << total_flops << ","
                  << std::fixed << std::setprecision(4) << flops_per_cycle << ","
                  << std::fixed << std::setprecision(0) << total_bytes << ","
                  << std::fixed << std::setprecision(4) << bytes_per_cycle << ","
                  << (rep_pass ? "true" : "false") << ","
                  << std::scientific << std::setprecision(4) << rep_max_diff
                  << std::endl;
    }
    std::cout << "--- CSV DATA END ---" << std::endl;

    mapsFree(d_X);
    mapsFree(d_W);
    mapsFree(d_Y);
    mapsFree(d_cycles);
    mapsStreamDestroy(stream);

    return 0;
}
