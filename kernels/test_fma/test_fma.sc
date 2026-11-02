#include "scale.h"
#include "maps_runtime_api.h"
#include "aurora.h"
#include "aurora_s-intrin.h"
#include "scale/barrier"
#include "cooperative_groups.h"

#if defined(__aurora_s__) || defined(__confidence__)
#include "llvm/tick.h"
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>

extern "C" __device__ KERNEL_M void mpu_add_u32(unsigned int length);

// Mode 0: In-register accumulation (acc += a * b, loop over N)
__device__ KERNEL_M void mpu_fma_reg(v16fp32 *c, const v16fp32 *a, const v16fp32 *b, unsigned N) {
    v16fp32 acc = *c;
    v16fp32 va = *a;
    v16fp32 vb = *b;
    for (unsigned i = 0; i < N; ++i) {
        acc += va * vb;
    }
    *c = acc;
}

// Mode 1: Chained dependency (acc = acc * a + b)
__device__ KERNEL_M void mpu_fma_chain(v16fp32 *c, const v16fp32 *a, const v16fp32 *b, unsigned N) {
    v16fp32 acc = *c;
    v16fp32 va = *a;
    v16fp32 vb = *b;
    for (unsigned i = 0; i < N; ++i) {
        acc = acc * va + vb;
    }
    *c = acc;
}

// Mode 2: DM buffer accumulation (matching GEMM inner loop)
__device__ KERNEL_M void mpu_fma_dm(v16fp32 *c, const v16fp32 *a, const v16fp32 *b, unsigned N) {
    for (unsigned i = 0; i < N; ++i) {
        c[0] += a[0] * b[0];
    }
}

// Mode 3: Hand-written VLIW assembly (vendor official_demo mpu_add_u32)
__device__ KERNEL_S void call_mpu_add_u32_asm(unsigned int input1_addr, unsigned int input2_addr, unsigned int output_addr, unsigned int N) {
    v16u32 params_input1 = {
        input1_addr, input1_addr, input1_addr, input1_addr,
        0x0000010<<2, 0x100<<2, 0x0, 0x0,
        ((N) << 16) | (N), 0x01000100, 0x0, 0xffffffff,
        0x00010000, 0x0, 0x00000f0c, 0xffff0000
    };
    v16u32 params_input2 = {
        input2_addr, input2_addr, input2_addr, input2_addr,
        0x0000010<<2, 0x100<<2, 0x0, 0x0,
        ((N) << 16) | (N), 0x01000100, 0x0, 0xffffffff,
        0x00010000, 0x0, 0x00000f0c, 0xffff0000
    };
    v16u32 params_output = {
        output_addr, output_addr, output_addr, output_addr,
        0x0000010<<2, 0x100<<2, 0x0, 0x0,
        ((N) << 16) | (N), 0x01000100, 0x0, 0xffffffff,
        0x00010000, 0x0, 0x00000f0c, 0xffff0000
    };

    __aurora_s_setMReg_var(0, *(v16u32*)&params_input1);
    __aurora_s_setMReg_var(1, *(v16u32*)&params_input2);
    __aurora_s_setMReg_var(2, *(v16u32*)&params_output);

    mpu_add_u32(N);
    __aurora_s_getStatB();
    __aurora_s_fenceI();
}

__global__ void test_fma_kernel(
    __DDR float *c_out,
    const __DDR float *a_in,
    const __DDR float *b_in,
    unsigned N,
    unsigned mode,
    __DDR unsigned long long *cycles_out)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid != 0) return;

    __mem0__ v16fp32 va_buf[1024] __attribute__((aligned(64)));
    __mem1__ v16fp32 vb_buf[1024] __attribute__((aligned(64)));
    __mem2__ v16fp32 vc_buf[1024] __attribute__((aligned(64)));

    auto block = cooperative_groups::this_thread_block();
    using barrier_t = scale::barrier<scale::thread_scope_block>;
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        init(&bar, 1);
    }
    __syncthreads();

    scale::memcpy_async(block, (float*)va_buf, a_in, scale::aligned_size_t<4>(sizeof(v16fp32)), bar);
    scale::memcpy_async(block, (float*)vb_buf, b_in, scale::aligned_size_t<4>(sizeof(v16fp32)), bar);
    scale::memcpy_async(block, (float*)vc_buf, c_out, scale::aligned_size_t<4>(sizeof(v16fp32)), bar);
    bar.arrive_and_wait();

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t0 = tick();
#endif

    if (mode == 0) {
        mpu_fma_reg(vc_buf, va_buf, vb_buf, N);
    } else if (mode == 1) {
        mpu_fma_chain(vc_buf, va_buf, vb_buf, N);
    } else if (mode == 2) {
        mpu_fma_dm(vc_buf, va_buf, vb_buf, N);
    } else if (mode == 3) {
        call_mpu_add_u32_asm((unsigned int)((uintptr_t)va_buf),
                             (unsigned int)((uintptr_t)vb_buf),
                             (unsigned int)((uintptr_t)vc_buf),
                             N);
    }
    MPU_SYNC();

#if defined(__aurora_s__) || defined(__confidence__)
    unsigned long long t1 = tick();
    cycles_out[0] = t1 - t0;
#endif

    scale::memcpy_async(block, c_out, (float*)vc_buf, scale::aligned_size_t<4>(sizeof(v16fp32)), bar);
    bar.arrive_and_wait();
}

int main(int argc, char** argv) {
    unsigned N = 100;
    unsigned mode = 0;
    int repeats = 3;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) mode = std::atoi(argv[2]);
    if (argc >= 4) repeats = std::atoi(argv[3]);

    float *h_a = (float*)malloc(16 * sizeof(float));
    float *h_b = (float*)malloc(16 * sizeof(float));
    float *h_c = (float*)malloc(16 * sizeof(float));

    for (int i = 0; i < 16; ++i) {
        h_a[i] = 1.0001f;
        h_b[i] = 0.0001f;
        h_c[i] = 0.0f;
    }

    float *d_a, *d_b, *d_c;
    unsigned long long *d_cycles;
    mapsMalloc((void**)&d_a, 16 * sizeof(float));
    mapsMalloc((void**)&d_b, 16 * sizeof(float));
    mapsMalloc((void**)&d_c, 16 * sizeof(float));
    mapsMalloc((void**)&d_cycles, sizeof(unsigned long long));

    mapsMemcpy(d_a, h_a, 16 * sizeof(float), mapsMemcpyHostToDevice);
    mapsMemcpy(d_b, h_b, 16 * sizeof(float), mapsMemcpyHostToDevice);
    mapsMemcpy(d_c, h_c, 16 * sizeof(float), mapsMemcpyHostToDevice);

    for (int r = 0; r < repeats; ++r) {
        dim3 grid(1, 1, 1);
        dim3 block(1, 1, 1);
        test_fma_kernel<<<grid, block>>>((__DDR float*)d_c, (__DDR float*)d_a, (__DDR float*)d_b, N, mode, (__DDR unsigned long long*)d_cycles);
        mapsDeviceSynchronize();

        unsigned long long dev_cycles = 0;
        mapsMemcpy(&dev_cycles, d_cycles, sizeof(unsigned long long), mapsMemcpyDeviceToHost);
        std::cout << "RESULT: N=" << N << " mode=" << mode << " repeat=" << (r + 1) << " cycles=" << dev_cycles << std::endl;
    }

    mapsFree(d_a);
    mapsFree(d_b);
    mapsFree(d_c);
    mapsFree(d_cycles);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
