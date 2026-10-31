#include "scale.h"
#include "scale_types.h"
#include "aurora.h"
#include "maps_runtime_api.h"
#include "maps_runtime_api_private.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "maops.h"

#define N 1024  // 向量大小
int N_ = 16;

extern __global__ void simple_copy32_stress(__DDR unsigned *dst, __DDR unsigned *src);
extern __global__ void simple_copy32(__DDR unsigned *dst, __DDR unsigned *src);
extern __global__ __system_cache__ void printf_debug(__DDR unsigned *dst, __DDR unsigned *src);
extern __global__ void vaddu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void vaddu32_nomemcpy(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void vmulu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num, __DDR unsigned long long *addr);
extern __global__ void vsubu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void vsubu32_nomemcpy(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void print();
extern __global__ void print_another();
extern __global__ void print_stress();
extern __global__ void vtanhfp32(__DDR float* dst, const __DDR float* src,
                          const unsigned int N_per_thread);
extern __global__ void profiler_devbuf();
extern __global__ void print1();

int memcpy_test()
{
    int ret = 0;
    unsigned *h_A, *h_B;
    unsigned *d_A, *d_B;
    uint64_t d_A_pa, d_B_pa;
    int fail_flag = false;
    int count = 8000 * 4;
    int i = 0;
    mapsHostAlloc((void**)&h_A, count * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_B, count * sizeof(unsigned), 0);
    mapsMalloc((void**)&d_A, count * sizeof(unsigned));
    mapsMalloc((void**)&d_B, count * sizeof(unsigned));
    
    // count = 8000 * 4
    for (i = 0; i < count; i++) {
      h_A[i] = i;
      h_B[i] = 0;
    }

    mapsMemcpy(d_A, h_A, count * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(h_B, d_A, count * sizeof(unsigned), mapsMemcpyDeviceToHost);
    printf("check h[h] after memcpyh2d & memcpyd2h:\n");
    for (i = 0; i < count; i++) {
      if (h_A[i] != h_B[i]) {
        printf("Error on %d: %u[%u]\n", i, h_B[i], h_A[i]);
        fail_flag = true;
      }
    }

    mapsMemcpy(d_A, h_A, count * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, d_A, count * sizeof(unsigned), mapsMemcpyDeviceToDevice);
    mapsMemcpy(h_B, d_B, count * sizeof(unsigned), mapsMemcpyDeviceToHost);
    for (i = 0; i < count; i++) {
      if (h_A[i] != h_B[i]) {
        printf("Error on %d: %u[%u]\n", i, h_B[i], h_A[i]);
        fail_flag = true;
      }
    }

    // count = 3456 * 7
    count = 3456 * 7;
    for (i = 0; i < count; i++) {
      h_A[i] = i + 1;
      h_B[i] = 0;
    }

    mapsMemcpy(d_A, h_A, count * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(h_B, d_A, count * sizeof(unsigned), mapsMemcpyDeviceToHost);
    printf("check h[h] after memcpyh2d & memcpyd2h:\n");
    for (i = 0; i < count; i++) {
      if (h_A[i] != h_B[i]) {
        printf("Error on %d: %u[%u]\n", i, h_B[i], h_A[i]);
        fail_flag = true;
      }
    }

    mapsMemcpy(d_A, h_A, count * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, d_A, count * sizeof(unsigned), mapsMemcpyDeviceToDevice);
    mapsMemcpy(h_B, d_B, count * sizeof(unsigned), mapsMemcpyDeviceToHost);
    for (i = 0; i < count; i++) {
      if (h_A[i] != h_B[i]) {
        printf("Error on %d: %u[%u]\n", i, h_B[i], h_A[i]);
        fail_flag = true;
      }
    }

    if (fail_flag)
      printf("Test FAIL\n");
    else {
      printf("Test PASS\n");
      ret++;
    }
    
    mapsFreeHost(h_A);
    mapsFreeHost(h_B);
    mapsFree(d_A);
    mapsFree(d_B);
    return ret;
}

int print_test()
{
    printf("[TEST BEGIN] test print kernel\n");
    dim3 blockDim(4, 1, 1);
    dim3 gridDim(4, 1, 1);
    print<<<gridDim, blockDim>>>();
    mapsDeviceSynchronize();
    print_another<<<gridDim, blockDim>>>();
    mapsDeviceSynchronize();
    return 0;
}

int spu_test()
{
    printf("[TEST BEGIN] test only spu kernel (print from 1 to 16)\n");
    int ret = 0;
    unsigned *h_A, *h_B;
    unsigned *d_A, *d_B;
    uint64_t d_A_pa, d_B_pa;
    int count = 4;
    int fail_flag = false;
    int i = 0;
    mapsHostAlloc((void**)&h_A, count * count * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_B, count * count * sizeof(unsigned), 0);
    mapsMalloc((void**)&d_A, count * count * sizeof(unsigned));
    mapsMalloc((void**)&d_B, count * count * sizeof(unsigned));
    
    for (i = 0; i < count * count; i++) {
        h_A[i] = i + 1;
        h_B[i] = 0;
    }
    mapsMemcpy(d_A, h_A, count * count * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, h_B, count * count * sizeof(unsigned), mapsMemcpyHostToDevice);

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    dim3 blockDim(count, 1, 1);
    dim3 gridDim(count, 1, 1);
    simple_copy32<<<gridDim, blockDim>>>((__DDR unsigned*)d_B_pa, (__DDR unsigned*)d_A_pa);
    mapsDeviceSynchronize();

    mapsMemcpy(h_B, d_B, count * count * sizeof(unsigned), mapsMemcpyDeviceToHost);
    for (i = 0; i < count * count; i++) {
        printf("%d ", h_B[i]);
        if (h_A[i] != h_B[i]) {
            fail_flag = true;
            break;
        }
    }
    printf("\n");

    if (fail_flag) {
        printf("Test FAIL\n");
        ret++;
    } else {
        printf("Test PASS\n");
    }

    mapsFreeHost(h_A);
    mapsFreeHost(h_B);
    mapsFree(d_A);
    mapsFree(d_B);
    return ret;
}

int stress_spu_test()
{
    dim3 blockDim(4, 1, 1);
    dim3 gridDim(4, 1, 1);
    int i = 0, j = 0;
    for (i = 0; i < 40; i++) {
        print<<<gridDim, blockDim>>>();
        print_another<<<gridDim, blockDim>>>();
        if ((i + 1) % 10 == 0) {
            mapsDeviceSynchronize();
        }
    }

    mapsDeviceSynchronize();
    return 0;
}

int mpu_test()
{
    printf("[TEST BEGIN] test complex kernel with spu & mpu\n");
    int ret = 0;
    unsigned *h_A, *h_B, *h_C, *h_AA=NULL, *h_CC=NULL ;
    unsigned *d_A, *d_B, *d_C, *d_D=NULL;
    uint64_t d_A_pa, d_B_pa, d_C_pa;

    mapsHostAlloc((void**)&h_A, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_B, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_C, N * sizeof(unsigned), 0);

    printf("[TEST] test memcpy\n");
    h_AA = (unsigned*)malloc(N * sizeof(unsigned));
    if (NULL == h_AA) {
        std::cout << "malloc failed" << std::endl;
        return 1;
    }
    memset(h_AA, 0x0, N * sizeof(unsigned));

    h_CC = (unsigned*)malloc(N * sizeof(unsigned));
    if (NULL == h_CC) {
        std::cout << "malloc failed" << std::endl;
        return 1;
    }
    memset(h_CC, 0x0, N * sizeof(unsigned));

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<unsigned>(i);
        h_B[i] = static_cast<unsigned>(i);
        h_AA[i] = h_A[i];
    }

    mapsMalloc((void**)&d_A, N * sizeof(unsigned));
    mapsMalloc((void**)&d_B, N * sizeof(unsigned));
    mapsMalloc((void**)&d_C, N * sizeof(unsigned));
    mapsMalloc((void**)&d_D, N * sizeof(unsigned));

    mapsMemcpy(d_A, h_A, N * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, h_B, N * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_D, h_AA, N * sizeof(unsigned), mapsMemcpyHostToDevice);

    for (int i=0; i < N; i+=100) {
        if (d_A[i] != d_D[i]) {
            std::cout << "!!!  HTOD test fail" << std::endl;
        }
        else {
            std::cout << "***  HTOD test succeed" << std::endl;
        }
    }

    printf("[TEST] test vaddu32\n");
    std::cout << "d_A: " << std::hex << d_A << std::endl;
    std::cout << "d_B: " << std::hex << d_B << std::endl;
    std::cout << "d_C: " << std::hex << d_C << std::endl;

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    mapsMemobjGetPhysicalAddress(d_C, &d_C_pa);

    dim3 blockDim(16, 1, 1);
    dim3 gridDim(4, 1, 1);

    std::cout << "d_A_pa: " << std::hex << d_A_pa << std::endl;
    std::cout << "d_B_pa: " << std::hex << d_B_pa << std::endl;
    std::cout << "d_C_pa: " << std::hex << d_C_pa << std::endl;

    //vaddu32_nomemcpy<<<gridDim, blockDim>>>((__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_B_pa, N_);
    vaddu32<<<gridDim, blockDim>>>((__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_B_pa, N_);
    mapsDeviceSynchronize();

    mapsMemcpy(h_C, d_C, N * sizeof(unsigned), mapsMemcpyDeviceToHost);
    mapsMemcpy(h_CC, d_C, N * sizeof(unsigned), mapsMemcpyDeviceToHost);

    for (int i=0; i < N; i+=100) {
        if (h_C[i] != h_CC[i]) {
            std::cout << "!!!  DTOH test fail" << std::endl;
        }
        else {
            std::cout << "***  DTOH test succeed" << std::endl;
        }
    }

    unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    threads = threads * gridDim.x * gridDim.y * gridDim.z;
    std::cout << "Add Result: " << std::endl;
    for (int j = 0; j < N_*threads/16; j++) {
      for (int i = 0; i < 16; ++i)
        std::cout << h_C[i + j*16] << " ";
      std::cout << std::endl;
    }
    bool fail_flag = false;
    for (int k = 0; k < N_*threads; k++) {
      if (h_C[k] != h_A[k] + h_B[k]) {
        fail_flag = true;
        break;
      }
    }

    if (!fail_flag)
      std::cout << "Test PASS" << std::endl;
    else
      std::cout << "Test FAIL" << std::endl;

    if (fail_flag)
        ret++;
#if 1
    printf("[TEST] test vsub32\n");
    for (int i = 0; i < N; ++i) {
        h_A[i] = 0;
        d_A[i] = 0;
    }
    //vsubu32_nomemcpy<<<gridDim, blockDim>>>((__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_B_pa, N_);
    vsubu32<<<gridDim, blockDim>>>((__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_B_pa, N_);
    mapsDeviceSynchronize();
    mapsMemcpy(h_A, d_A, N * sizeof(unsigned), mapsMemcpyDeviceToHost);

    std::cout << "Sub Result: " << std::endl;
    for (int j = 0; j < N_*threads/16; j++) {
      for (int i = 0; i < 16; ++i)
        std::cout << h_A[i + j*16] << " ";
      std::cout << std::endl;
    }
    fail_flag = false;
    for (int k = 0; k < N_*threads; k++) {
      if (h_A[k] != h_C[k] - h_B[k]) {
        fail_flag = true;
        break;
      }
    }
    if (!fail_flag)
      std::cout << "Test PASS" << std::endl;
    else
      std::cout << "Test FAIL" << std::endl;

    if (fail_flag)
        ret++;
#endif
    mapsFreeHost(h_A);
    mapsFreeHost(h_B);
    mapsFreeHost(h_C);
    free(h_AA);
    free(h_CC);
    mapsFree(d_A);
    mapsFree(d_B);
    mapsFree(d_C);
    mapsFree(d_D);
    return ret;
}

int multi_stream_test()
{
    printf("[TEST BEGIN] test multi stream kernel launch\n");
    int ret = 0;
    int fail_flag = false;
    unsigned *h_A, *h_B, *h_C, *h_D, *h_E;
    unsigned *d_A, *d_B, *d_C, *d_D, *d_E;
    uint64_t d_A_pa, d_B_pa, d_C_pa, d_D_pa, d_E_pa;
    mapsStream_t stream0, stream1;
    mapsEvent_t event0, event1;
    int i = 0;
    dim3 blockDim(16, 1, 1);
    dim3 gridDim(4, 1, 1);
    dim3 printfBlockDim(8, 1, 1);
    dim3 printfGridDim(2, 1, 1);
    unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    threads = threads * gridDim.x * gridDim.y * gridDim.z;

    mapsStreamCreate(&stream0);
    mapsStreamCreate(&stream1);
    mapsEventCreate(&event0);
    mapsEventCreate(&event1);

    mapsHostAlloc((void**)&h_A, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_B, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_C, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_D, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_E, N * sizeof(unsigned), 0);

    for (i = 0; i < N; i++) {
        h_A[i] = static_cast<unsigned>(i + 1);
        h_B[i] = static_cast<unsigned>(1);
        h_C[i] = h_D[i] = h_E[i] = static_cast<unsigned>(0);
    }

    mapsMalloc((void**)&d_A, N * sizeof(unsigned));
    mapsMalloc((void**)&d_B, N * sizeof(unsigned));
    mapsMalloc((void**)&d_C, N * sizeof(unsigned));
    mapsMalloc((void**)&d_D, N * sizeof(unsigned));
    mapsMalloc((void**)&d_E, N * sizeof(unsigned));

    mapsMemcpy(d_A, h_A, N * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, h_B, N * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_C, h_C, N * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_D, h_D, N * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_E, h_E, N * sizeof(unsigned), mapsMemcpyHostToDevice);

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    mapsMemobjGetPhysicalAddress(d_C, &d_C_pa);
    mapsMemobjGetPhysicalAddress(d_D, &d_D_pa);
    mapsMemobjGetPhysicalAddress(d_E, &d_E_pa);

    vsubu32<<<gridDim, blockDim, 0, (scaleStream_t)stream0>>>((__DDR unsigned*)d_D_pa, (__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_B_pa, N_);
    vaddu32<<<gridDim, blockDim, 0, (scaleStream_t)stream0>>>((__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_B_pa, N_);
    mapsEventRecord(event0, stream0);

    print<<<printfGridDim, printfBlockDim, 0, (scaleStream_t)stream1>>>();
    mapsStreamWaitEvent(stream1, event0, 0);
    vaddu32<<<gridDim, blockDim, 0, (scaleStream_t)stream1>>>((__DDR unsigned*)d_D_pa, (__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_D_pa, N_);
    mapsStreamSynchronize(stream1);
    mapsMemcpy(h_D, d_D, N * sizeof(unsigned), mapsMemcpyDeviceToHost);

    std::cout << "Result: " << std::endl;
    for (int j = 0; j < N_*threads/16; j++) {
      for (int i = 0; i < 16; ++i)
        std::cout << h_D[i + j*16] << " ";
      std::cout << std::endl;
    }
    fail_flag = false;
    for (int k = 0; k < N_*threads; k++) {
      if (h_D[k] != h_A[k] + h_B[k] + h_A[k] - h_B[k]) {
        fail_flag = true;
        break;
      }
    }
    if (!fail_flag)
      std::cout << "Test PASS" << std::endl;
    else
      std::cout << "Test FAIL" << std::endl;

    if (fail_flag)
        ret++;

    mapsFreeHost(h_A);
    mapsFreeHost(h_B);
    mapsFreeHost(h_C);
    mapsFreeHost(h_D);
    mapsFreeHost(h_E);
    mapsFree(d_A);
    mapsFree(d_B);
    mapsFree(d_C);
    mapsFree(d_D);
    mapsFree(d_E);
    mapsEventDestroy(event0);
    mapsEventDestroy(event1);
    mapsStreamDestroy(stream0);
    mapsStreamDestroy(stream1);
    return ret;
}

int c_mode_test()
{
    int ret = 0;
    bool fail_flag = false;
    unsigned *h_A, *h_B, *h_C;
    unsigned *d_A, *d_B, *d_C;
    uint64_t d_A_pa, d_B_pa, d_C_pa;
    uint64_t *h_D;
    uint64_t *d_D;
    uint64_t d_D_pa;
    mapsStream_t stream;

    mapsHostAlloc((void**)&h_A, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_B, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_C, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_D, N * sizeof(unsigned long long), 0);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<unsigned>(i);
        h_B[i] = static_cast<unsigned>(i);
    }

    mapsMalloc((void**)&d_A, N * sizeof(unsigned));
    mapsMalloc((void**)&d_B, N * sizeof(unsigned));
    mapsMalloc((void**)&d_C, N * sizeof(unsigned));
    mapsMalloc((void**)&d_D, N * sizeof(unsigned long long));

    mapsMemcpy(d_A, h_A, N * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, h_B, N * sizeof(unsigned), mapsMemcpyHostToDevice);

    std::cout << "d_A: " << std::hex << d_A << std::endl;
    std::cout << "d_B: " << std::hex << d_B << std::endl;
    std::cout << "d_C: " << std::hex << d_C << std::endl;
    std::cout << "d_D: " << std::hex << d_D << std::endl;

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    mapsMemobjGetPhysicalAddress(d_C, &d_C_pa);
    mapsMemobjGetPhysicalAddress(d_D, &d_D_pa);

    std::cout << "d_A_pa: " << std::hex << d_A_pa << std::endl;
    std::cout << "d_B_pa: " << std::hex << d_B_pa << std::endl;
    std::cout << "d_C_pa: " << std::hex << d_C_pa << std::endl;
    std::cout << "d_D_pa: " << std::hex << d_D_pa << std::endl;

    mapsLaunchConfig_t config = {0};
    config.blockDim.x = 4;
    config.blockDim.y = 1;
    config.blockDim.z = 1;

    config.gridDim.x = 4;
    config.gridDim.y = 1;
    config.gridDim.z = 1;

    mapsLaunchAttribute attr[2];
    attr[0].id = mapsLaunchAttributeConcurrencyThread;
    attr[0].val.concurrencyThread = 1;
    attr[1].id = mapsLaunchAttributeSmq;
    attr[1].val.smqEnable = 1;

    config.attrs = attr;
    config.numAttrs = 2;
    void* args[5] = {&d_C_pa, &d_A_pa, &d_B_pa, &N_, &d_D_pa};
    mapsLaunchKernelExc(&config, (void*)&vmulu32, args);

    mapsMemcpy(h_C, d_C, N * sizeof(unsigned), mapsMemcpyDeviceToHost);
    mapsMemcpy(h_D, d_D, N * sizeof(unsigned), mapsMemcpyDeviceToHost);

    unsigned threads = config.blockDim.x * config.blockDim.y * config.blockDim.z;
    threads = threads * config.gridDim.x * config.gridDim.y * config.gridDim.z;
    std::cout << "Mul Result: " << std::endl;
    for (int j = 0; j < N_*threads/16; j++) {
      for (int i = 0; i < 16; ++i)
        std::cout << h_C[i + j*16] << " ";
      std::cout << std::endl;
    }
    std::cout << "address of local_src0: " << std::endl;
    for (int i = 0; i < 4; ++i)
      std::cout << h_D[i] << ", ";
    std::cout << std::endl;
    std::cout << "address of local_src1: " << std::endl;
    for (int i = 4; i < 8; ++i)
      std::cout << h_D[i] << ", ";
    std::cout << std::endl;
    std::cout << "address of local_dst: " << std::endl;
    for (int i = 8; i < 12; ++i)
      std::cout << h_D[i] << ", ";
    std::cout << std::endl;

    for (int k = 0; k < N_*threads; k++) {
      if (h_C[k] != h_A[k] * h_B[k]) {
        fail_flag = true;
        break;
      }
    }
    if (!fail_flag)
      std::cout << "Test PASS" << std::endl;
    else
      std::cout << "Test FAIL" << std::endl;

    if (fail_flag)
        ret++;

    mapsFreeHost(h_A);
    mapsFreeHost(h_B);
    mapsFreeHost(h_C);
    mapsFreeHost(h_D);
    mapsFree(d_A);
    mapsFree(d_B);
    mapsFree(d_C);
    mapsFree(d_D);
    return ret;
}

void tanh_fp32(const void* src, void* dst, const unsigned int element_count) {
  const unsigned int block_x_dim = 4;
  dim3 blockDim(block_x_dim, 1, 1);

  const unsigned int N_per_thread = element_count / block_x_dim;
  const unsigned int N_per_block = N_per_thread * blockDim.x;
  int grid_x = (element_count + N_per_block - 1) / N_per_block;
  dim3 gridDim(grid_x, 1, 1);

  vtanhfp32<<<gridDim, blockDim>>>((__DDR float*)dst, (const __DDR float*)src, N_per_thread);
  mapsDeviceSynchronize();
}

int tanh_test(float *input, float *output, double eps_1, double eps_2, int max_ulp,
                  maopsDataType_t MAOPS_DTYPE, int total_element_size) {
  float *h_input, *h_output;
  float *d_input, *d_output;
  uint64_t d_input_pa, d_output_pa;
  bool fail_flag = false;
  int ret = 0;

  mapsHostAlloc((void**)&h_input, total_element_size * sizeof(float), 0);
  mapsHostAlloc((void**)&h_output, total_element_size * sizeof(float), 0);
  mapsMalloc((void**)&d_input, total_element_size * sizeof(float));
  mapsMalloc((void**)&d_output, total_element_size * sizeof(float));

  for (int i = 0; i < total_element_size; ++i) {
    h_input[i] = input[i];
  }

  mapsMemcpy(d_input, h_input, total_element_size * sizeof(float), mapsMemcpyHostToDevice);

  mapsMemobjGetPhysicalAddress(d_input, &d_input_pa);
  mapsMemobjGetPhysicalAddress(d_output, &d_output_pa);
  tanh_fp32((void*)d_input_pa, (void*)d_output_pa, total_element_size);

  mapsMemcpy(h_output, d_output, total_element_size * sizeof(float), mapsMemcpyDeviceToHost);

  for (int i = 0; i < total_element_size; ++i) {
    printf("%f [check: %f]\n", h_output[i], output[i]);
    if (abs(h_output[i] - output[i]) / output[i] > eps_1 || abs(h_output[i] - output[i]) > eps_2) {
      fail_flag = true;
    }
  }

  if (!fail_flag)
    std::cout << "Test PASS" << std::endl;
  else
    std::cout << "Test FAIL" << std::endl;
  
  if (fail_flag)
    ret++;

  mapsFreeHost(h_input);
  mapsFreeHost(h_output);
  mapsFree(d_input);
  mapsFree(d_output);
  return ret;
}

int mpu_stress_test() {
    int ret = 0;
    unsigned *h_A, *h_B, *h_C;
    unsigned *d_A, *d_B, *d_C;
    uint64_t d_A_pa, d_B_pa, d_C_pa;
    mapsStream_t stream = nullptr;

    mapsHostAlloc((void**)&h_A, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_B, N * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_C, N * sizeof(unsigned), 0);
    mapsMalloc((void**)&d_A, N * sizeof(unsigned));
    mapsMalloc((void**)&d_B, N * sizeof(unsigned));
    mapsMalloc((void**)&d_C, N * sizeof(unsigned));
    mapsStreamCreate(&stream);

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<unsigned>(i);
        h_B[i] = static_cast<unsigned>(i);
    }
    mapsMemcpyAsync(d_A, h_A, N * sizeof(unsigned), mapsMemcpyHostToDevice, stream);
    mapsMemcpyAsync(d_B, h_B, N * sizeof(unsigned), mapsMemcpyHostToDevice, stream);
    mapsDeviceSynchronize();

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    mapsMemobjGetPhysicalAddress(d_C, &d_C_pa);

    dim3 blockDim(16, 1, 1);
    dim3 gridDim(4, 1, 1);

    for (int i = 0; i < 80; i++) {
        vaddu32<<<gridDim, blockDim, 0, (scaleStream_t)stream>>>((__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_B_pa, N_);
    }
    mapsDeviceSynchronize();

    mapsMemcpyAsync(h_C, d_C, N * sizeof(unsigned), mapsMemcpyDeviceToHost, stream);
    mapsDeviceSynchronize();

    unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    threads = threads * gridDim.x * gridDim.y * gridDim.z;
    std::cout << "Add Result: " << std::endl;
    for (int j = 0; j < N_*threads/16; j++) {
      for (int i = 0; i < 16; ++i)
        std::cout << h_C[i + j*16] << " ";
      std::cout << std::endl;
    }
    bool fail_flag = false;
    for (int k = 0; k < N_*threads; k++) {
      if (h_C[k] != h_A[k] + h_B[k]) {
        fail_flag = true;
        break;
      }
    }

    if (!fail_flag)
      std::cout << "Test PASS" << std::endl;
    else
      std::cout << "Test FAIL" << std::endl;

    if (fail_flag)
        ret++;
    
    mapsFreeHost(h_A);
    mapsFreeHost(h_B);
    mapsFreeHost(h_C);
    mapsFree(d_A);
    mapsFree(d_B);
    mapsFree(d_C);
    mapsStreamDestroy(stream);
    return ret;
}

#undef N
#define N 7168
int memcpy_debug() {
  int B = 2, M_Max = 4096;
  uint64_t* matC_dev = nullptr;
#define USE_MAPS_ALLOC
#ifdef USE_MAPS_ALLOC
  mapsMalloc((void**)&matC_dev, (B * N * M_Max * sizeof(uint16_t)));
#endif
  uint16_t* matC = (uint16_t*)malloc(B * N * M_Max * sizeof(uint16_t));
  float* matC_unquant_dev = nullptr;
#ifdef USE_MAPS_ALLOC
  mapsMalloc((void**)&matC_unquant_dev, (B * N * M_Max * sizeof(float)));
#endif
  float* matC_unquant = (float*)malloc(B * N * M_Max * sizeof(float));

  printf("current matC ptr=0x%llx, matC_unquant ptr=0x%llx\n", (unsigned long long)matC, (unsigned long long)matC_unquant);

  double denominator = 0;
  double sim = 0;
  int M = M_Max;
  for (int b = 0; b < B; ++b){
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        uint16_t x_i = matC[b * M * N + m * N + n];
        uint32_t x_i32 = (x_i << 16);
        float x;
        x = x_i32;
        
        float y = matC_unquant[b * M * N + m * N + n];
        denominator += (x * x + y * y);
        sim += 2 * x * y;
      }
      
      if (m % 16 == 0) {
        printf("(b, m): (%d, %d)\n", b, m);
      }
    }
  }
#ifdef USE_MAPS_ALLOC
  mapsFree(matC_dev);
#endif
  free(matC);
#ifdef USE_MAPS_ALLOC
  mapsFree(matC_unquant_dev);
#endif
  free(matC_unquant);

  return 0;
}

int profiler_devbuf_test()
{
    printf("[TEST BEGIN] test profiler devbuf kernel\n");

    int r = 0;
    const size_t sz = sizeof(unsigned);
    unsigned *d_A, *d_B;
    uint64_t d_A_pa, d_B_pa;
    unsigned *h_ptr = (unsigned*)malloc(sz);
    *h_ptr = 0;

    mapsStream_t stream = nullptr;
    mapsStreamCreate(&stream);

    mapsMalloc((void**)&d_A, sz);
    mapsMalloc((void**)&d_B, sz);
    mapsMemcpy(d_A, h_ptr, sz, mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, h_ptr, sz, mapsMemcpyHostToDevice);

    dim3 blockDim(4, 1, 1);
    dim3 gridDim(1, 1, 1);

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsSetProfilerBufferEx((void*)d_A_pa, stream);
    profiler_devbuf<<<gridDim, blockDim, 0, (scaleStream_t)stream>>>();
    mapsDeviceSynchronize();

    mapsMemcpy(h_ptr, d_A, sz, mapsMemcpyDeviceToHost);
    if (*h_ptr == 0) {
        printf("FAIL after kernel: d_A = 0x%x to h_ptr = 0x%x\n", *d_A, *h_ptr);
        r++;
    }

    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    mapsSetProfilerBufferEx((void*)d_B_pa, stream);
    profiler_devbuf<<<gridDim, blockDim, 0, (scaleStream_t)stream>>>();
    mapsDeviceSynchronize();

    *h_ptr = 0;
    mapsMemcpy(h_ptr, d_B, sz, mapsMemcpyDeviceToHost);
    if (*h_ptr == 0) {
        printf("FAIL after kernel: d_B = 0x%x to h_ptr = 0x%x\n", *d_B, *h_ptr);
        r++;
    }

    mapsSetProfilerBufferEx(nullptr, stream);
    profiler_devbuf<<<gridDim, blockDim, 0, (scaleStream_t)stream>>>();
    mapsDeviceSynchronize();

    mapsFree(d_A);
    mapsFree(d_B);
    free(h_ptr);
    mapsStreamDestroy(stream);
    return r;
}

int setenv_test()
{
    int ret = 0;
    dim3 blockDim(4, 1, 1);
    dim3 gridDim(4, 1, 1);
    mapsSetEnv("MAPS_KERNEL_REPEATITION_COUNT", "2");
    print_stress<<<gridDim, blockDim>>>();
    mapsSetEnv("MAPS_KERNEL_REPEATITION_COUNT", "1");
    print_stress<<<gridDim, blockDim>>>();

    return 0;
}

int noc_cacheable_test() {
    printf("[TEST BEGIN] test printf!!!\n");
    int ret = 0;
    unsigned *h_A, *h_B;
    unsigned *d_A, *d_B;
    uint64_t d_A_pa, d_B_pa;
    int count = 4;
    int fail_flag = false;
    int i = 0;
    mapsHostAlloc((void**)&h_A, count * count * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_B, count * count * sizeof(unsigned), 0);
    mapsMalloc((void**)&d_A, count * count * sizeof(unsigned));
    mapsMalloc((void**)&d_B, count * count * sizeof(unsigned));
    
    for (i = 0; i < count * count; i++) {
        h_A[i] = count * count - i;
        h_B[i] = 0;
    }
    mapsMemcpy(d_A, h_A, count * count * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, h_B, count * count * sizeof(unsigned), mapsMemcpyHostToDevice);

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    dim3 blockDim(count, 1, 1);
    dim3 gridDim(count, 1, 1);
    printf_debug<<<gridDim, blockDim>>>((__DDR unsigned*)d_B_pa, (__DDR unsigned*)d_A_pa);
    mapsDeviceSynchronize();

    printf("expect: 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n");
    printf("result: ");
    mapsMemcpy(h_B, d_B, count * count * sizeof(unsigned), mapsMemcpyDeviceToHost);
    for (i = 0; i < count * count; i++) {
        printf("%d ", h_B[i]);
    }
    printf("\n");

    mapsFreeHost(h_A);
    mapsFreeHost(h_B);
    mapsFree(d_A);
    mapsFree(d_B);
    return ret;
}

void print1_test() {
    dim3 blockDim(16, 1, 1);
    dim3 gridDim(1, 1, 1);
    print1<<<gridDim, blockDim>>>();
}

int maps_func_get_params_test() {
    int ret = 0;
    bool result_flag = true;
    const char* funcName = nullptr;
    size_t paramCount = 0;
    static size_t paramOffsets_vaddu32[] = {0x0, 0x8, 0x10, 0x18};
    static size_t paramSizes_vaddu32[] = {8, 8, 8, 4};

    ret = mapsFuncGetName(&funcName, (void*)&vaddu32);
    if(funcName != nullptr) {
        printf("maps_func_get_params_test vaddu32 name: %s\n", funcName);
    } else {
        result_flag = false;
    }

    ret += mapsFuncGetParamCount((void*)&vaddu32, &paramCount);
    if(paramCount == 4) {
        printf("maps_func_get_params_test vaddu32 paramCount: %zu\n", paramCount);
    } else {
        result_flag = false;
    }

    for(size_t paramIndex = 0; paramIndex < paramCount; ++paramIndex) {
        size_t paramOffset = 0, paramSize = 0;
        ret += mapsFuncGetParamInfo((void*)&vaddu32, paramIndex, &paramOffset, &paramSize);
        if((paramOffset == paramOffsets_vaddu32[paramIndex]) && (paramSize == paramSizes_vaddu32[paramIndex])) {
            printf("maps_func_get_params_test vaddu32 paramIndex: %zu, paramOffset: 0x%zx, paramSize: %zu\n", paramIndex, paramOffset, paramSize);
        } else {
            result_flag = false;
			break;
        }
    }
	
    if (result_flag) {
        printf("maps_func_get_params_test Test PASS\n");
    } else {
        printf("maps_func_get_params_test Test FAIL\n");
    }
    return ret;
}

extern __device__ unsigned* global_src1; 
extern __device__ unsigned* global_src2;
extern __device__ unsigned* global_dst1;
extern __device__ unsigned* global_dst2;

/*
    Test Content
    1. use symbol: global_src1 -> dev1 -> host1-> global_dst1
    2. use address: global_src2 -> host2 -> dev2 -> global_dst2
*/
int memcpy_symbol_test() {
    int ret = 0;

    mapsInit(0);

    mapsStream_t stream = nullptr;
    mapsStreamCreate(&stream);

    unsigned* dev_global_src1;
    unsigned* dev_global_src2;
    unsigned* dev_global_dst1;
    unsigned* dev_global_dst2;

    size_t size1;
    size_t size2;
    
    mapsGetSymbolAddress((void**) &dev_global_src1, MAPS_SYMBOL(global_src1));
    mapsGetSymbolAddress((void**) &dev_global_dst1, MAPS_SYMBOL(global_dst1));
    mapsGetSymbolAddress((void**) &dev_global_src2, MAPS_SYMBOL(global_src2));
    mapsGetSymbolAddress((void**) &dev_global_dst2, MAPS_SYMBOL(global_dst2));

    mapsGetSymbolSize((size_t*) &size1, MAPS_SYMBOL(global_src1));
    mapsGetSymbolSize((size_t*) &size2, MAPS_SYMBOL(global_src2));

    unsigned* dev1;
    unsigned* host1;
    unsigned* dev2;
    unsigned* host2;
    mapsMalloc((void**) &dev1, size1);
    mapsMalloc((void**) &dev2, size2);
    mapsMallocHost((void**) &host1, size1);
    mapsMallocHost((void**) &host2, size2);

    mapsMemcpyFromSymbolAsync(dev1, MAPS_SYMBOL(global_src1), size1, 0x0, mapsMemcpyDeviceToDevice, stream);
    mapsMemcpyAsync(host1, dev1, size1, mapsMemcpyDeviceToHost, stream);
    mapsMemcpyToSymbolAsync(MAPS_SYMBOL(global_dst1), host1, size1, 0x0, mapsMemcpyHostToDevice, stream);
    mapsStreamSynchronize(stream);

    for (int i = 0x0; i < size1 / sizeof(unsigned int); i++) {
        printf("0x%x ", dev_global_dst1[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }

        if (dev_global_src1[i] != dev_global_dst1[i]) {
            ret++;
            break;
        }
    }

    if (ret != 0) {
        printf("Test FAILED!!!\n");
        goto  exit_free;
    } else {
        printf("Test PASSED!!!\n");
    }

    mapsMemcpy(host2, dev_global_src2, size2, mapsMemcpyDeviceToHost);
    mapsMemcpy(dev2, host2, size2, mapsMemcpyHostToDevice);
    mapsMemcpy(dev_global_dst2, dev2, size2, mapsMemcpyDeviceToDevice);
    mapsDeviceSynchronize();

    for (int i = 0x0; i < size2 / sizeof(unsigned int); i++) {
        printf("0x%x ", dev_global_dst2[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }

        if (dev_global_src2[i] != dev_global_dst2[i]) {
            ret++;
            break;
        }
    }

    if (ret != 0) {
        printf("Test FAILED!!!\n");
        goto  exit_free;
    } else {
        printf("Test PASSED!!!\n");
    }

exit_free:
    mapsFree(dev1);
    mapsFree(dev2);
    mapsFreeHost(host1);
    mapsFreeHost(host2);
    mapsStreamDestroy(stream);
    return ret;
}

// test metrics mask & set env & MAPS_DUMP_KERNEL_LAUNCH_ARGS_ENABLE
int metrics_mask_test() {
    unsigned *d_A[4], *d_B[4];
    uint64_t d_A_pa[4], d_B_pa[4];
    void* args[4][2];

    mapsStream_t stream = nullptr;
    mapsStreamCreate(&stream);

    for (int i = 0; i < 4; i++) {
        mapsMalloc((void**)&d_A[i], 1 * 4 * sizeof(unsigned));
        mapsMalloc((void**)&d_B[i], 1 * 4 * sizeof(unsigned));

        mapsMemobjGetPhysicalAddress(d_A[i], &d_A_pa[i]);
        mapsMemobjGetPhysicalAddress(d_B[i], &d_B_pa[i]);

        args[i][0] = &d_B_pa[i];
        args[i][1] = &d_A_pa[i];
    }

    for (int i = 0; i < 4; i++) {
        printf("args[%d]: 0x%llx, 0x%llx\n", i, d_B_pa[i], d_A_pa[i]);
    }

    mapsLaunchAttribute attr;
    attr.id = mapsLaunchAttributeMetricsMask;
    attr.val.metricsMask = 0xF0F0ULL;

    mapsLaunchConfig_t config = {0};
    config.gridDim = dim3(1, 1, 1);
    config.blockDim = dim3(4, 1, 1);
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;

    mapsSetEnv("MAPS_DUMP_KERNEL_LAUNCH_ARGS_ENABLE", "true");

    int device_id = 0;
    uint32_t apc_mask;
    mapsGetDevice(&device_id);
    mapsGetApcMask(device_id, &apc_mask);
    uint32_t g_perf_kids[4] = {1, 1, 1, 1};
    uint32_t g_perf_wgids[4] = {0, 1, 2, 3};
    uint32_t g_perf_warpids[4] = {0, 0, 0, 0};
    mapsStartKernelProfiling(stream, apc_mask, g_perf_kids, g_perf_wgids, g_perf_warpids);
    mapsLaunchKernelExc(&config, (void*) simple_copy32_stress, args[0]);
    mapsStopKernelProfiling(stream, apc_mask);

    mapsLaunchKernelExc(&config, (void*) simple_copy32_stress, args[1]);
    mapsLaunchKernelExc(&config, (void*) simple_copy32_stress, args[2]);
    mapsStreamSynchronize(stream);

    mapsSetEnv("MAPS_DUMP_KERNEL_LAUNCH_ARGS_ENABLE", "false");

    mapsLaunchKernelExc(&config, (void*) simple_copy32_stress, args[3]);
    mapsStreamSynchronize(stream);

    for (int i = 0; i < 4; i++) {
        mapsFree(d_A[i]);
        mapsFree(d_B[i]);
    }

    mapsStreamDestroy(stream);

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc >= 2) { // for project S test
      const char* scale_arch = argv[1];
      if (std::strcmp(scale_arch, "prj_s") == 0) {
        int ret = 0;
        print1_test();
        ret += spu_test();
        ret += print_test();
        ret += profiler_devbuf_test();
        ret += maps_func_get_params_test();
        return ret;
      }
    }

    int ret = 0;

    //ret += memcpy_debug();
    //ret += memcpy_debug();
    //ret += memcpy_test();

    char* noc_cacheable_env = NULL;
    if ((noc_cacheable_env = getenv("MAPS_SPU_NOC_CACHEABLE_MEM_ENABLE")) != NULL &&
      strncmp(noc_cacheable_env, "true", 4) == 0) {
        ret += noc_cacheable_test();
        ret += noc_cacheable_test();
    }

    metrics_mask_test();

    ret += memcpy_symbol_test();
    if (ret != 0) {
        return -1;
    }

    print1_test();
    print1_test();
    
    // ret += setenv_test();

    ret += print_test();
    ret += print_test();

    ret += profiler_devbuf_test();
    ret += profiler_devbuf_test();

    ret += spu_test();
    ret += spu_test();

    // ret += stress_spu_test();

    ret += mpu_test();
    ret += mpu_test();

    // ret += mpu_stress_test();
    
    ret += multi_stream_test();
    ret += multi_stream_test();

    ret += c_mode_test();
    ret += c_mode_test();

    float input[32] = {3.56732, 2.12093, 5.18625, 4.68237, 8.77736, 4.69974, 3.38108, 0.202188, 6.45849, 5.69385, 9.7192, 6.02247, 3.55115, 2.19619, 5.6324, 5.56633, 9.31013, 3.49562, 2.89118, 9.63941, 6.40283, 4.26379, 9.87123, 2.51698, 3.47896, 9.53527, 7.42723, 4.64005, 6.45479, 9.66829, 2.17871, 7.38042};
    float output[32] = {0.998407, 0.971646, 0.999937, 0.999829, 1, 0.999834, 0.997689, 0.199478, 0.999995, 0.999977, 1, 0.999988, 0.998355, 0.97556, 0.999974, 0.999971, 1, 0.998162, 0.993856, 1, 0.999995, 0.999604, 1, 0.987058, 0.9981, 1, 0.999999, 0.999813, 0.999995, 1, 0.974701, 0.999999};
    ret += tanh_test(input, output, 1e-3, 1e-3, 1, MAOPS_DTYPE_FLOAT, 32);
    ret += tanh_test(input, output, 1e-3, 1e-3, 1, MAOPS_DTYPE_FLOAT, 32);

    ret += maps_func_get_params_test();
    return ret;
}

