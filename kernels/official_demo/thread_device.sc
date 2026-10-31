#include "scale.h"
#include "scale_types.h"
#include "aurora.h"
#include "maps_runtime_api.h"
#include <stdlib.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string.h>
#include <chrono>
#include "pthread.h"
#include <vector>

extern __global__ void simple_copy32(__DDR unsigned *dst, __DDR unsigned *src);
extern __global__ void print();

static void* threadFunc(void* args)
{
    int dev = *(int*) args;

    mapsSetDevice(dev);
    if (dev == 0) {
        printf("[TEST BEGIN] run print on device: 0\n");
        dim3 blockDim(4, 1, 1);
        dim3 gridDim(4, 1, 1);
        print<<<gridDim, blockDim>>>();
        mapsDeviceSynchronize();
    } else {
        printf("[TEST BEGIN] run simple_copy32 on device: %d\n", dev);
        int ret = 0;
        unsigned *h_A, *h_B;
        unsigned *d_A, *d_B;
        uint64_t d_A_pa, d_B_pa;
        int count = 4;
        int fail_flag = false;
        int i = 0;
        mapsHostAlloc((void **)&h_A, count * count * sizeof(unsigned), 0);
        mapsHostAlloc((void **)&h_B, count * count * sizeof(unsigned), 0);
        mapsMalloc((void **)&d_A, count * count * sizeof(unsigned));
        mapsMalloc((void **)&d_B, count * count * sizeof(unsigned));

        for (i = 0; i < count * count; i++)
        {
            h_A[i] = i + 1;
            h_B[i] = 0;
        }
        mapsMemcpy(d_A, h_A, count * count * sizeof(unsigned), mapsMemcpyHostToDevice);
        mapsMemcpy(d_B, h_B, count * count * sizeof(unsigned), mapsMemcpyHostToDevice);

        mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
        mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
        dim3 blockDim(count, 1, 1);
        dim3 gridDim(count, 1, 1);
        simple_copy32<<<gridDim, blockDim>>>((__DDR unsigned *)d_B_pa, (__DDR unsigned *)d_A_pa);
        mapsDeviceSynchronize();

        mapsMemcpy(h_B, d_B, count * count * sizeof(unsigned), mapsMemcpyDeviceToHost);
        for (i = 0; i < count * count; i++)
        {
            printf("%d ", h_B[i]);
            if (h_A[i] != h_B[i])
            {
                fail_flag = true;
                break;
            }
        }
        printf("\n");

        if (fail_flag)
        {
            printf("Test FAIL\n");
            ret++;
        }
        else
        {
            printf("Test PASS\n");
        }

        mapsFreeHost(h_A);
        mapsFreeHost(h_B);
        mapsFree(d_A);
        mapsFree(d_B);
    }
#ifndef __SCALE_ARCH__
    pthread_exit(NULL);
#endif
}


typedef struct thread_info_st {
    pthread_t tid;
    int dev;
} thread_info_t;

int main(int argc, char *argv[])
{
    int devCnt = 0;
    int n = 0;
    mapsGetDeviceCount(&devCnt);

    std::vector<thread_info_t*> thread_infos;
    for (int i = 0; i < devCnt; i++) {
        thread_info_t* thread_info = (thread_info_t*) malloc(sizeof(thread_info_t));
        if (thread_info == nullptr) {
            printf("FAILED to create new thread info!!!");
            return -1;
        }

        thread_info->dev = i;
#ifndef __SCALE_ARCH__
        if (pthread_create(&thread_info->tid, NULL, threadFunc, (void*) &thread_info->dev)) {
            printf("FAILED to create pthread!!!");
            return -1;
        }
#endif

        thread_infos.push_back(thread_info);
    }

    for (int i = 0; i < devCnt; i++) {
#ifndef __SCALE_ARCH__
        pthread_join(thread_infos[i]->tid, NULL);
#endif
        free(thread_infos[i]);
    }
    thread_infos.clear();

    return 0;
}


