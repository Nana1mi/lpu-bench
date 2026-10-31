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

extern __global__ void vaddu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void test_counter(__DDR unsigned *dst, unsigned num);

#define TEST_NUM 1024
#define TEST_SIZE (TEST_NUM * sizeof(uint32_t))
int N_ = 16;
#define MAPS_RESOURCE_NUM 4
typedef enum
{
  TEN_K_CASE = 0,
  BASE_ADD_CASE,
  ALL_CASE
} test_case_t;

mapsStream_t streams[MAPS_RESOURCE_NUM];
mapsEvent_t events[MAPS_RESOURCE_NUM];
uint32_t* host_mems[MAPS_RESOURCE_NUM];
uint32_t* device_mems[MAPS_RESOURCE_NUM];
uint64_t device_mem_pas[MAPS_RESOURCE_NUM];

int test_device_counter() {
    dim3 blockDim(16, 1, 1);
    dim3 gridDim(4, 1, 1);

    for (int i = 0; i < TEST_NUM; i++) {
        device_mems[0][i] = 0;
    }
    test_counter<<<gridDim, blockDim, 0, (scaleStream_t)streams[0]>>>((__DDR unsigned*)device_mem_pas[0], 16*4);
    mapsMemcpyAsync(host_mems[0], device_mems[0], TEST_SIZE, mapsMemcpyDeviceToHost, streams[0]);
    mapsStreamSynchronize(streams[0]);
    for (int i = 0; i < 16*4; i++) {
        printf("%d ", host_mems[0][i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    printf("Test PASS\n");
    return 0;
}

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

static int set_environment(void)
{
  if (setenv("ROC_AQL_QUEUE_SIZE", "524288", 0) != 0) { //1024 * 512, 512K (524288)
    fprintf(stderr, "setenv failed: %s\n", strerror(errno));
    return -1;
  }

  if (setenv("MAPS_KERNEL_REPEATITION_COUNT", "10000", 0) != 0) {
    fprintf(stderr, "setenv failed: %s\n", strerror(errno));
    return -1;
  }

  if (setenv("MAPS_KICK_QUEUE_AT_MARKER", "true", 0) != 0) {
    fprintf(stderr, "setenv failed: %s\n", strerror(errno));
    return -1;
  }

  return 0;
}

int main(int argc, char *argv[]) {
    int ret = 0;
    if (set_environment() != 0) {
      return -1;
    }
    for (int i = 0; i < MAPS_RESOURCE_NUM; i++) {
        mapsMalloc((void**)(&device_mems[i]), TEST_SIZE);
        mapsMallocHost((void**)(&host_mems[i]), TEST_SIZE);
        mapsStreamCreate(&streams[i]);
        mapsEventCreate(&events[i]);
        mapsMemobjGetPhysicalAddress(device_mems[i], &device_mem_pas[i]);
    }

    test_case_t test_case_option = TEN_K_CASE;
    if (argc > 1) {
      test_case_option = (test_case_t)atoi(argv[1]);
      printf("test_case: '%d'\n", (int)test_case_option);
    }

    if (TEN_K_CASE == test_case_option) {
      printf("----- run test device counter case\n");
      ret = test_device_counter();
    }
    else if (BASE_ADD_CASE == test_case_option) {
      printf("----- run test basic add case\n");
      ret = test2();
    }
    else if (ALL_CASE == test_case_option) {
      printf("----- run all case\n");
      ret = test2();
      if (!ret) {
        ret = test_device_counter();
      }
    }
    else {
      printf("!!!!! error! invalid case number\n");
      goto free_exit;
    }

    if (ret != 0) {
        printf("Test Fail\n");
        goto free_exit;
    }

free_exit:
    for (int i = 0; i < MAPS_RESOURCE_NUM; i++) {
        mapsFree(device_mems[i]);
        mapsFreeHost(host_mems[i]);
        mapsStreamDestroy(streams[i]);
        mapsEventDestroy(events[i]);
    }

    return ret;
}
