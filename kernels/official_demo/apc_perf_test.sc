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
#include <sys/time.h>
#include "maops.h"

extern __global__ void simple_copy32_stress(__DDR unsigned *dst, __DDR unsigned *src);

void apc_perf_test(int grid_num, int stream_num)
{
    const int block_num = 4;
    unsigned *h_A[32], *h_B[32];
    unsigned *d_A[32], *d_B[32];
    uint64_t d_A_pa[32], d_B_pa[32];
    mapsStream_t stream[32];

    struct timeval start, end;

    for (int i = 0; i < stream_num; i++) {
        mapsStreamCreate(&stream[i]);

        mapsHostAlloc((void**)&h_A[i], grid_num * block_num * sizeof(unsigned), 0);
        mapsHostAlloc((void**)&h_B[i], grid_num * block_num * sizeof(unsigned), 0);
        mapsMalloc((void**)&d_A[i], grid_num * block_num * sizeof(unsigned));
        mapsMalloc((void**)&d_B[i], grid_num * block_num * sizeof(unsigned));

        mapsMemobjGetPhysicalAddress(d_A[i], &d_A_pa[i]);
        mapsMemobjGetPhysicalAddress(d_B[i], &d_B_pa[i]);
    }

    gettimeofday(&start, NULL);
    for (int i = 0; i < stream_num; i++) {
        mapsLaunchConfig_t config = {0};
        config.blockDim.x = block_num;
        config.blockDim.y = 1;
        config.blockDim.z = 1;

        config.gridDim.x = grid_num;
        config.gridDim.y = 1;
        config.gridDim.z = 1;

        config.stream = stream[i];

        mapsLaunchAttribute attr[3];
        attr[0].id = mapsLaunchAttributeConcurrencyThread;
        attr[0].val.concurrencyThread = 1;
        attr[1].id = mapsLaunchAttributeApcMask;
        attr[1].val.apcMask = (((0x1 << grid_num) - 1) << (grid_num * i));
        attr[2].id = mapsLaunchAttributeClusterDimension;
        attr[2].val.clusterDim.x = config.gridDim.x;
        attr[2].val.clusterDim.y = config.gridDim.y;
        attr[2].val.clusterDim.z = config.gridDim.z;

        config.attrs = attr;
        if (grid_num == 1) {
            config.numAttrs = 2;
        } else {
            config.numAttrs = 3;
        }
        
        void* args[2] = {&d_B_pa[i], &d_A_pa[i]};
        mapsLaunchKernelExc(&config, (void*)&simple_copy32_stress, args);
    }

    mapsDeviceSynchronize();
    gettimeofday(&end, NULL);

    printf("stream number: %2d, grid numner: %2d, block number: %2d, use %llu us\n", 
        stream_num, grid_num, block_num, 
        (end.tv_sec - start.tv_sec) * 1000000ULL + (end.tv_usec - start.tv_usec));

    for (int i = 0; i < stream_num; i++) {
        mapsFreeHost(h_A[i]);
        mapsFreeHost(h_B[i]);
        mapsFree(d_A[i]);
        mapsFree(d_B[i]);
        mapsStreamDestroy(stream[i]);
    }
}

int main(int argc, char *argv[])
{
    printf("=Warm up======================================================================\n");
    apc_perf_test(1, 1);
    printf("==============================================================================\n");
    printf("=Result=======================================================================\n");

    apc_perf_test(1, 1);
    apc_perf_test(1, 4);
    apc_perf_test(1, 5);
    apc_perf_test(1, 8);
    apc_perf_test(1, 13);
    apc_perf_test(1, 16);
    printf("==============================================================================\n");
}