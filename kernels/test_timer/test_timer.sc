#include "scale.h"
#include "maps_runtime_api.h"
#include "aurora.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#if defined(__aurora_s__) || defined(__confidence__)
#include "llvm/tick.h"
#endif

#include "timer.h"

// -------------------------------------------------------------
// Device Kernels (SPU & MPU)
// -------------------------------------------------------------
__device__ KERNEL_M void mpu_workload(v16fp32 *dst, const v16fp32 *src, unsigned n_vec) {
    for (unsigned i = 0; i < n_vec; i++) {
        dst[i] = dst[i] * src[i] + src[i];
    }
}

__global__ void timer_kernel(__DDR unsigned long long *dev_cycles_out, unsigned loop_count) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __mem0__ v16fp32 local_a[16] __attribute__((aligned(64)));
    __mem1__ v16fp32 local_b[16] __attribute__((aligned(64)));

    for (int i = 0; i < 16; i++) {
        local_a[i] = (v16fp32){1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        local_b[i] = (v16fp32){0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f,
                               0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    }

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t_start = tick();
    for (unsigned k = 0; k < loop_count; k++) {
        mpu_workload(local_a, local_b, 16);
        MPU_SYNC();
    }
    unsigned long long t_end = tick();

    if (tid == 0) {
        dev_cycles_out[0] = t_end - t_start;
    }
#endif
}

// -------------------------------------------------------------
// Host Verification Program
// -------------------------------------------------------------
int main(int argc, char** argv) {
    std::cout << "[TIMING_BENCHMARK] Starting timing validation suite..." << std::endl;

    unsigned long long *h_cycles = nullptr;
    unsigned long long *d_cycles = nullptr;
    uint64_t d_cycles_pa = 0;

    mapsHostAlloc((void**)&h_cycles, sizeof(unsigned long long), 0);
    mapsMalloc((void**)&d_cycles, sizeof(unsigned long long));
    mapsMemobjGetPhysicalAddress(d_cycles, &d_cycles_pa);

    mapsStream_t stream;
    mapsStreamCreate(&stream);
    HostEventTimer timer(stream);

    // Warm-up run
    dim3 blockDim(4, 1, 1);
    dim3 gridDim(1, 1, 1);
    timer_kernel<<<gridDim, blockDim, 0, stream>>>((__DDR unsigned long long*)d_cycles_pa, 5);
    mapsStreamSynchronize(stream);

    // Workload sweep: 1x (10), 2x (20), 4x (40), 8x (80)
    std::vector<unsigned> workloads = {10, 20, 40, 80};
    const int repeats = 3;

    std::cout << "--- CSV DATA START ---" << std::endl;
    std::cout << "workload_mult,loop_count,repeat,sim_ms,sim_cycles,device_cycles,wall_ms" << std::endl;

    for (size_t w_idx = 0; w_idx < workloads.size(); ++w_idx) {
        unsigned loop = workloads[w_idx];
        int mult = 1 << w_idx;

        for (int r = 0; r < repeats; ++r) {
            h_cycles[0] = 0;
            mapsMemcpyAsync(d_cycles, h_cycles, sizeof(unsigned long long), mapsMemcpyHostToDevice, stream);

            timer.start();
            timer_kernel<<<gridDim, blockDim, 0, stream>>>((__DDR unsigned long long*)d_cycles_pa, loop);
            timer.stop();

            mapsMemcpyAsync(h_cycles, d_cycles, sizeof(unsigned long long), mapsMemcpyDeviceToHost, stream);
            mapsStreamSynchronize(stream);

            TimingResult res = timer.get_result(h_cycles[0]);

            std::cout << mult << ","
                      << loop << ","
                      << r << ","
                      << std::fixed << std::setprecision(4) << res.sim_ms << ","
                      << res.sim_cycles << ","
                      << res.device_cycles << ","
                      << std::fixed << std::setprecision(2) << res.wall_clock_ms << std::endl;
        }
    }
    std::cout << "--- CSV DATA END ---" << std::endl;

    mapsStreamDestroy(stream);
    mapsFree(d_cycles);
    mapsFreeHost(h_cycles);

    std::cout << "[TIMING_BENCHMARK] Timing validation complete." << std::endl;
    return 0;
}
