#include "scale.h"
#include "scale_types.h"
#include "aurora.h"
#include "maps_runtime_api.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <string.h>

#include "scale.h"
#include "scale_types.h"
#include "aurora.h"
#include "maps_runtime_api.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <string.h>

extern __global__ void simple_copy32(__DDR unsigned *dst, __DDR unsigned *src);
extern __global__ void vaddu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void vaddu32_nomemcpy(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void vmulu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num, __DDR unsigned long long *addr);
extern __global__ void vsubu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void print();

#define TEST_NUM 1024
#define TEST_SIZE (TEST_NUM * sizeof(uint32_t))
int N_ = 16;

mapsStream_t streams[70];
mapsEvent_t events[70];
uint32_t* host_mems[70];
uint32_t* device_mems[70];
uint64_t device_mem_pas[70];

int test2() {
    dim3 blockDim(16, 1, 1);
    dim3 gridDim(4, 1, 1);

    for (int i = 0; i < TEST_NUM; i++) {
        host_mems[0][i] = i;
        host_mems[1][i] = i + 2;
        host_mems[2][i] = 0;
    }
    mapsMemcpyAsync(device_mems[0], host_mems[0], TEST_SIZE, mapsMemcpyHostToDevice, streams[0]);
    mapsMemcpyAsync(device_mems[1], host_mems[1], TEST_SIZE, mapsMemcpyHostToDevice, streams[0]);
    vaddu32<<<gridDim, blockDim, 0, (scaleStream_t)streams[0]>>>((__DDR unsigned*)device_mem_pas[2],
        (__DDR unsigned*)device_mem_pas[0], (__DDR unsigned*)device_mem_pas[1], N_);
    mapsMemcpyAsync(host_mems[2], device_mems[2], TEST_SIZE, mapsMemcpyDeviceToHost, streams[0]);
    mapsStreamSynchronize(streams[0]);
    for (int i = 0; i < TEST_NUM; i++) {
        printf("%d ", host_mems[2][i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
        if (host_mems[2][i] != host_mems[0][i] + host_mems[1][i]) {
            return 1;
        }
    }
    printf("Test PASS\n");
    return 0;
}

int test3() {
    dim3 blockDim(16, 1, 1);
    dim3 gridDim(4, 1, 1);

    for (int i = 0; i < TEST_NUM; i++) {
        host_mems[0][i] = i;
        host_mems[1][i] = i + 2;
        host_mems[2][i] = 0;
    }
    mapsMemcpyAsync(device_mems[0], host_mems[0], TEST_SIZE, mapsMemcpyHostToDevice, streams[0]);
    mapsMemcpyAsync(device_mems[1], host_mems[1], TEST_SIZE, mapsMemcpyHostToDevice, streams[0]);
    vaddu32<<<gridDim, blockDim, 0, (scaleStream_t)streams[0]>>>((__DDR unsigned*)device_mem_pas[2],
        (__DDR unsigned*)device_mem_pas[0], (__DDR unsigned*)device_mem_pas[1], N_);
    vaddu32<<<gridDim, blockDim, 0, (scaleStream_t)streams[0]>>>((__DDR unsigned*)device_mem_pas[3],
        (__DDR unsigned*)device_mem_pas[0], (__DDR unsigned*)device_mem_pas[2], N_);
    mapsMemcpyAsync(host_mems[3], device_mems[3], TEST_SIZE, mapsMemcpyDeviceToHost, streams[0]);
    mapsStreamSynchronize(streams[0]);
    for (int i = 0; i < TEST_NUM; i++) {
        printf("%d ", host_mems[3][i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
        if (host_mems[3][i] != host_mems[0][i] + host_mems[1][i] + host_mems[0][i]) {
            return 1;
        }
    }
    printf("Test PASS\n");
    return 0;
}

int test_pitch(int pitch_size) {
    dim3 blockDim(16, 1, 1);
    dim3 gridDim(4, 1, 1);

    for (int k = 0; k < pitch_size; k++) {
        for (int i = 0; i < TEST_NUM; i++) {
            for (int j = 0; j < 4; j++) {
                host_mems[k * 4 + j][i] = i + j;
            }
        }

        for (int i = 0; i < 4; i++) {
            mapsMemcpyAsync(device_mems[k * 4 + i], host_mems[k * 4 + i], TEST_SIZE, mapsMemcpyHostToDevice, streams[k * 4 + i]);
        }

        for (int i = 0; i < 4; i++) {
            if (i != 0) {
                mapsStreamWaitEvent(streams[k * 4 + i], events[k * 4 + i - 1]);
                vaddu32<<<gridDim, blockDim, 0, (scaleStream_t)streams[k * 4 + i]>>>((__DDR unsigned*)device_mem_pas[k * 4 + i],
                    (__DDR unsigned*)device_mem_pas[k * 4 + i], (__DDR unsigned*)device_mem_pas[k * 4 + i - 1], N_);
            } else {
                vaddu32<<<gridDim, blockDim, 0, (scaleStream_t)streams[k * 4 + i]>>>((__DDR unsigned*)device_mem_pas[k * 4 + i],
                    (__DDR unsigned*)device_mem_pas[k * 4 + i], (__DDR unsigned*)device_mem_pas[k * 4 + i], N_);
            }
            mapsEventRecord(events[k * 4 + i], streams[k * 4 + i]);
            mapsMemcpyAsync(host_mems[k * 4 + i], device_mems[k * 4 + i], TEST_SIZE, mapsMemcpyDeviceToHost, streams[k * 4 + i]);
        }

        for (int i = 0; i < 4; i++) {
            mapsStreamSynchronize(streams[k * 4 + i]);
            printf("host mem %d:\n", k * 4 + i);
            for (int j = 0; j < TEST_NUM; j++) {
                printf("%d ", host_mems[k * 4 + i][j]);
                if ((j + 1) % 16 == 0) {
                    printf("\n");
                }
                if (host_mems[k * 4 + i][j] != j * (i + 2) + i * (i + 1) / 2) {
                    return 1;
                }
            }
        }
        printf("Test PASS\n");
    }
    return 0;
}

int test12() {
    return test_pitch(1);
}

int test13() {
    return test_pitch(8);
}

int test14() {
    return test_pitch(16);
}

int main(int argc, char *argv[]) {
    int ret = 0;
    for (int i = 0; i < 70; i++) {
        mapsMalloc((void**)(&device_mems[i]), TEST_SIZE);
        mapsMallocHost((void**)(&host_mems[i]), TEST_SIZE);
        mapsStreamCreate(&streams[i]);
        mapsEventCreate(&events[i]);
        mapsMemobjGetPhysicalAddress(device_mems[i], &device_mem_pas[i]);
    }

    //ret += test2();
    if (ret > 0) {
        printf("Test Fail\n");
        goto free_exit;
    }
    //ret += test3();
    if (ret > 0) {
        printf("Test Fail\n");
        goto free_exit;
    }
    //ret += test12();
    if (ret > 0) {
        printf("Test Fail\n");
        goto free_exit;
    }
    ret += test13();
    if (ret > 0) {
        printf("Test Fail\n");
        goto free_exit;
    }

free_exit:
    for (int i = 0; i < 70; i++) {
        mapsFree(device_mems[i]);
        mapsFreeHost(host_mems[i]);
        mapsStreamDestroy(streams[i]);
        mapsEventDestroy(events[i]);
    }

    return ret;
}