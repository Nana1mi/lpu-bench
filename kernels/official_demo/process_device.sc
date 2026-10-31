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
#include <sys/wait.h>
#include <signal.h>

extern __global__ void simple_copy32(__DDR unsigned *dst, __DDR unsigned *src);
extern __global__ void print();

static void fatherProcessFunc() {
    int ret = 0;
    mapsSetDevice(0);
    printf("[TEST BEGIN] run print on device: 0\n");
    dim3 blockDim(4, 1, 1);
    dim3 gridDim(4, 1, 1);
    print<<<gridDim, blockDim>>>();
    mapsDeviceSynchronize();
}

static void childProcessFunc(int processId)
{
    int dev = processId >> 1;
    mapsSetDevice(dev);
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

/*
    Test content
    process 0 - device 0 father process
    process 1 - device 0 child process
    process 2 - device 1 child process
    process 3 - device 1 child process 
*/
int main(int argc, char *argv[])
{
    int processCnt = 4;
    int n = 0;

    for (int i = 1; i < processCnt; i++) {
        pid_t pid = fork();
        switch (pid) {
            case -1:
                exit(-1);
            case 0:
                childProcessFunc(i);
                return 0;
            default:
                break;
        }
    }

    fatherProcessFunc();
    return 0;
}


