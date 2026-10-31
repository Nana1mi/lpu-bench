#include "scale.h"
#include "scale_types.h"
#include "aurora.h"
#include "maps_runtime_api.h"
#include "maps_runtime_api_private.h"
#include <stdlib.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string.h>
#include <chrono>

#define ARRAY_SIZE(arr)                 (sizeof(arr) / sizeof((arr)[0]))

extern __global__ void simple_copy32(__DDR unsigned *dst, __DDR unsigned *src);
extern __global__ void vaddu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void vaddu32_nomemcpy(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void vaddu32_asm(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void vmulu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num, __DDR unsigned long long *addr);
extern __global__ void vsubu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned num);
extern __global__ void print();

#define N 1024  // 向量大小
int N_ = 16;

static uint32_t g_apc_num = 0;

//input cmd para
static uint32_t g_grid_cnt = 4;
static uint32_t g_block_cnt = 4;
static uint32_t g_test_index = 0;
static uint32_t g_N = 1024;
static int g_test_loop = 1;
static uint32_t g_perf_kids[PROF_KERNEL_MAX] = {1, 1, 1, 1};
static uint32_t g_perf_wgids[PROF_KERNEL_MAX] = {0, 1, 2, 3};
static uint32_t g_perf_warpids[PROF_KERNEL_MAX] = {0, 0, 0, 0};
static uint32_t g_apc_mask = 1;

struct testCmd_s {
    int (*func)();
    const char *name;
    const char *helpStr;
};

int spu_test()
{
    printf("[TEST BEGIN] test only spu kernel (print from 1 to 16)\n");
    int ret = 0;
    unsigned *h_A, *h_B;
    unsigned *d_A, *d_B;
    uint64_t d_A_pa, d_B_pa;
    int fail_flag = false;
    int i = 0;
    mapsHostAlloc((void**)&h_A, g_grid_cnt * g_block_cnt * sizeof(unsigned), 0);
    mapsHostAlloc((void**)&h_B, g_grid_cnt * g_block_cnt * sizeof(unsigned), 0);
    mapsMalloc((void**)&d_A, g_grid_cnt * g_block_cnt * sizeof(unsigned));
    mapsMalloc((void**)&d_B, g_grid_cnt * g_block_cnt * sizeof(unsigned));

    for (i = 0; i < g_grid_cnt * g_block_cnt; i++) {
        h_A[i] = i + 1;
        h_B[i] = 0;
    }
    mapsMemcpy(d_A, h_A, g_grid_cnt * g_block_cnt * sizeof(unsigned), mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, h_B, g_grid_cnt * g_block_cnt * sizeof(unsigned), mapsMemcpyHostToDevice);

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    dim3 gridDim(g_grid_cnt, 1, 1);
    dim3 blockDim(g_block_cnt, 1, 1);
    simple_copy32<<<gridDim, blockDim>>>((__DDR unsigned*)d_B_pa, (__DDR unsigned*)d_A_pa);
    mapsDeviceSynchronize();

    mapsMemcpy(h_B, d_B, g_grid_cnt * g_block_cnt * sizeof(unsigned), mapsMemcpyDeviceToHost);
    for (i = 0; i < g_grid_cnt * g_block_cnt; i++) {
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

int mpu_test()
{
    printf("[TEST BEGIN] test complex kernel with spu & mpu\n");
    int ret = 0;
    unsigned *h_A, *h_B, *h_C, *h_AA=NULL, *h_CC=NULL ;
    unsigned *d_A, *d_B, *d_C, *d_D=NULL;
    uint64_t d_A_pa, d_B_pa, d_C_pa;
    uint32_t dataNum = g_N;
    uint32_t dataSize = dataNum * sizeof(unsigned);

    mapsHostAlloc((void**)&h_A, dataSize, 0);
    mapsHostAlloc((void**)&h_B, dataSize, 0);
    mapsHostAlloc((void**)&h_C, dataSize, 0);

    printf("[TEST] test memcpy\n");
    h_AA = (unsigned*)malloc(dataSize);
    if (NULL == h_AA) {
        std::cout << "malloc failed" << std::endl;
        return 1;
    }
    memset(h_AA, 0x0, dataSize);

    h_CC = (unsigned*)malloc(dataSize);
    if (NULL == h_CC) {
        std::cout << "malloc failed" << std::endl;
        return 1;
    }
    memset(h_CC, 0x0, dataSize);

    for (int i = 0; i < dataNum; ++i) {
        h_A[i] = static_cast<unsigned>(i);
        h_B[i] = static_cast<unsigned>(i);
        h_AA[i] = h_A[i];
    }

    mapsMalloc((void**)&d_A, dataSize);
    mapsMalloc((void**)&d_B, dataSize);
    mapsMalloc((void**)&d_C, dataSize);
    mapsMalloc((void**)&d_D, dataSize);

    mapsMemcpy(d_A, h_A, dataSize, mapsMemcpyHostToDevice);
    mapsMemcpy(d_B, h_B, dataSize, mapsMemcpyHostToDevice);
    mapsMemcpy(d_D, h_AA, dataSize, mapsMemcpyHostToDevice);

    for (int i=0; i < dataNum; i+=100) {
        if (d_A[i] != d_D[i]) {
            std::cout << "!!!  HTOD test fail" << std::endl;
        }
//        else {
//            std::cout << "***  HTOD test succeed" << std::endl;
//        }
    }

    printf("[TEST] test vaddu32\n");
    std::cout << "d_A: " << std::hex << d_A << std::endl;
    std::cout << "d_B: " << std::hex << d_B << std::endl;
    std::cout << "d_C: " << std::hex << d_C << std::endl;

    mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
    mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
    mapsMemobjGetPhysicalAddress(d_C, &d_C_pa);

    dim3 gridDim(g_grid_cnt, 1, 1);
    dim3 blockDim(g_block_cnt, 1, 1);

    std::cout << "d_A_pa: " << std::hex << d_A_pa << std::endl;
    std::cout << "d_B_pa: " << std::hex << d_B_pa << std::endl;
    std::cout << "d_C_pa: " << std::hex << d_C_pa << std::endl;

    unsigned threadsperblock = blockDim.x * blockDim.y * blockDim.z;
    unsigned totalthreads = threadsperblock * gridDim.x * gridDim.y * gridDim.z;
    unsigned numperthread = (dataNum + totalthreads - 1) / totalthreads;
    //vaddu32_nomemcpy<<<gridDim, blockDim>>>((__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_B_pa, numperthread);
    // vaddu32<<<gridDim, blockDim>>>((__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_B_pa, numperthread);
    vaddu32_asm<<<gridDim, blockDim>>>((__DDR unsigned*)d_C_pa, (__DDR unsigned*)d_A_pa, (__DDR unsigned*)d_B_pa, numperthread);
    mapsDeviceSynchronize();

    mapsMemcpy(h_C, d_C,  dataSize, mapsMemcpyDeviceToHost);
    mapsMemcpy(h_CC, d_C, dataSize, mapsMemcpyDeviceToHost);

    for (int i=0; i < dataNum; i+=100) {
        if (h_C[i] != h_CC[i]) {
            std::cout << "!!!  DTOH test fail" << std::endl;
        }
//        else {
//            std::cout << "***  DTOH test succeed" << std::endl;
//        }
    }

    unsigned threads = blockDim.x * blockDim.y * blockDim.z;
    threads = threads * gridDim.x * gridDim.y * gridDim.z;
    std::cout << "Add Result: " << std::endl;
    for (int j = 0; j < N_ * threads  / 16; j++) {
      for (int i = 0; i < 16; ++i)
        std::cout << h_C[i + j*16] << " ";
      std::cout << std::endl;
    }
    bool fail_flag = false;
    for (int k = 0; k < N_ * threads; k++) {
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
#if 0
    printf("[TEST] test vsub32\n");
    for (int i = 0; i < N; ++i) {
        h_A[i] = 0;
        d_A[i] = 0;
    }
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

struct testCmd_s g_test_cmd_tab[] = {
    {NULL, "all", "run all kernels."},
    {mpu_test, "mpu_test", "vaddu32_nomemcpy."},
    {spu_test, "spu_test", "simple_copy32."},
};

static int do_perf_kernel_test(uint32_t test_idx)
{
    if (test_idx < ARRAY_SIZE(g_test_cmd_tab) && g_test_cmd_tab[test_idx].func) {
        std::cout << "<<=========start do test kernel: " << g_test_cmd_tab[test_idx].name << " =========>>" << std::endl;
        g_test_cmd_tab[test_idx].func();
        mapsDeviceSynchronize();
        std::cout << "<<===========end do test kernel: " << g_test_cmd_tab[test_idx].name << " =========>>" << std::endl;
    }
    return 0;
}

static void gpuKernelRunThread() {
    uint32_t uiApcMask = g_apc_mask;
    struct apcKernelProfilingData startProf[32] = {0};
    struct apcKernelProfilingData endProf[32] = {0};
    static uint32_t test_idx = 1;
    mapsStream_t stream = NULL;
    uint64_t ns = 0;
    uint64_t cycle = 0;

    if (g_test_index != 0 && g_test_index < ARRAY_SIZE(g_test_cmd_tab)) {
        test_idx = g_test_index;
    } else if(test_idx >= ARRAY_SIZE(g_test_cmd_tab)) {
        test_idx = 1;
    }

    mapsStartKernelProfiling(stream, uiApcMask, g_perf_kids, g_perf_wgids, g_perf_warpids);
    mapsReadKernelProfiling(stream, uiApcMask, startProf);

    mapsRecordKernelPerfTime(stream, 1);
    do_perf_kernel_test(test_idx++);
    mapsGetKernelPerfTime(stream, &ns, &cycle);

    mapsReadKernelProfiling(stream, uiApcMask, endProf);
    mapsStopKernelProfiling(stream, uiApcMask);

    std::cout << std::endl << "++++++++++++++++Kernel:" << std::hex << g_perf_kids[0] << " profiling.++++++++++++++++" << std::endl;
    std::cout << std::endl << "++++++++++++++++Kernel[1] ns:" << ns << " cycle:" << cycle << std::endl;
    for (int i = 0; i < g_apc_num; i++) {
        if (uiApcMask & (1 << i)) {
            std::cout << std::endl << "APC:" << std::dec << i << std::endl;
            std::cout << "periodic_cnt:0x" << std::hex << endProf[i].periodic_cnt - startProf[i].periodic_cnt << std::endl;
            std::cout << std::hex;
            for (int k = 0; k < READY_TIMESTAMP_NUM; k++) {
                if (endProf[i].ready_stamp[k] & (0xfULL << 60)) {
                    uint32_t evid = (endProf[i].ready_stamp[k] >> 60) & 0xf;
                    uint64_t stamp = (endProf[i].ready_stamp[k] & ~(0xfULL << 60)) - startProf[i].periodic_cnt;
                    if (evid >= 5 && evid <= 8) {
                        std::cout << "Kid:0x" << g_perf_kids[evid - 5] << " WgId:0x" << g_perf_wgids[evid - 5] <<
                            " Warpid:0x" << g_perf_warpids[evid - 5] << " ready at:0x" << std::hex << stamp << std::endl;
                    }
                    else if (evid < 5)
                    {
                        std::cout << "Kid:0x" << g_perf_kids[evid - 1] << " WgId:0x" << g_perf_wgids[evid - 1] <<
                            " ready at:0x" << std::hex << stamp << std::endl;
                    }
                } else {
                    break;
                }
            }

            std::cout << "ape_specified_kernel_selected_warp_cnt:" << endProf[i].apc_profiling.ape_specified_kernel_selected_warp_cnt - startProf[i].apc_profiling.ape_specified_kernel_selected_warp_cnt << std::endl;
            std::cout << "ape_specified_kernel_stall_warp_cnt:" << endProf[i].apc_profiling.ape_specified_kernel_stall_warp_cnt - startProf[i].apc_profiling.ape_specified_kernel_stall_warp_cnt << std::endl;
            std::cout << "ape_warppool_empty_cnt:" << endProf[i].apc_profiling.ape_warppool_empty_cnt - startProf[i].apc_profiling.ape_warppool_empty_cnt << std::endl;
            std::cout << "ape_dma_starved_cnt:" << endProf[i].apc_profiling.ape_dma_starved_cnt - startProf[i].apc_profiling.ape_dma_starved_cnt << std::endl;
            std::cout << "ape_mpu_starved_cnt:" << endProf[i].apc_profiling.ape_mpu_starved_cnt - startProf[i].apc_profiling.ape_mpu_starved_cnt << std::endl;
            std::cout << "ape_spu_starved_cnt:" << endProf[i].apc_profiling.ape_spu_starved_cnt - startProf[i].apc_profiling.ape_spu_starved_cnt << std::endl;
            std::cout << "ape_dm_wr_rd_blocked_cnt:" << endProf[i].apc_profiling.ape_dm_wr_rd_blocked_cnt - startProf[i].apc_profiling.ape_dm_wr_rd_blocked_cnt << std::endl;
            std::cout << "warp_cnt:" << endProf[i].apc_profiling.warp_cnt - startProf[i].apc_profiling.warp_cnt << std::endl;
            std::cout << "ape_eligible_warp_cnt:" << endProf[i].apc_profiling.ape_eligible_warp_cnt - startProf[i].apc_profiling.ape_eligible_warp_cnt << std::endl;
            std::cout << "ape_fast_dispatch_stall_cnt:" << endProf[i].apc_profiling.ape_fast_dispatch_stall_cnt - startProf[i].apc_profiling.ape_fast_dispatch_stall_cnt << std::endl;
            std::cout << "ape_wg_barrier_stall_cnt:" << endProf[i].apc_profiling.ape_wg_barrier_stall_cnt - startProf[i].apc_profiling.ape_wg_barrier_stall_cnt << std::endl;
            std::cout << "ape_mpu_stall_cnt:" << endProf[i].apc_profiling.ape_mpu_stall_cnt - startProf[i].apc_profiling.ape_mpu_stall_cnt << std::endl;
            std::cout << "ape_dma_stall_cnt:" << endProf[i].apc_profiling.ape_dma_stall_cnt - startProf[i].apc_profiling.ape_dma_stall_cnt << std::endl;
            std::cout << "ape_cluster_barrier_stall_cnt:" << endProf[i].apc_profiling.ape_cluster_barrier_stall_cnt - startProf[i].apc_profiling.ape_cluster_barrier_stall_cnt << std::endl;
            std::cout << "ape_pre_pass_stall_cnt:" << endProf[i].apc_profiling.ape_pre_pass_stall_cnt - startProf[i].apc_profiling.ape_pre_pass_stall_cnt << std::endl;
            std::cout << "ape_dshm_stall_cnt:" << endProf[i].apc_profiling.ape_dshm_stall_cnt - startProf[i].apc_profiling.ape_dshm_stall_cnt << std::endl;
            std::cout << "ape_stalled_type0_cnt:" << endProf[i].apc_profiling.ape_stalled_type0_cnt - startProf[i].apc_profiling.ape_stalled_type0_cnt << std::endl;
            std::cout << "ape_fast_dispatch_resume_cnt:" << endProf[i].apc_profiling.ape_fast_dispatch_resume_cnt - startProf[i].apc_profiling.ape_fast_dispatch_resume_cnt << std::endl;
            std::cout << "ape_wg_barrier_resume_cnt:" << endProf[i].apc_profiling.ape_wg_barrier_resume_cnt - startProf[i].apc_profiling.ape_wg_barrier_resume_cnt << std::endl;
            std::cout << "ape_mpu_resume_cnt:" << endProf[i].apc_profiling.ape_mpu_resume_cnt - startProf[i].apc_profiling.ape_mpu_resume_cnt << std::endl;
            std::cout << "ape_dma_resume_cnt:" << endProf[i].apc_profiling.ape_dma_resume_cnt - startProf[i].apc_profiling.ape_dma_resume_cnt << std::endl;
            std::cout << "ape_cluster_barrier_resume_cnt:" << endProf[i].apc_profiling.ape_cluster_barrier_resume_cnt - startProf[i].apc_profiling.ape_cluster_barrier_resume_cnt << std::endl;
            std::cout << "ape_pre_pass_resume_cnt:" << endProf[i].apc_profiling.ape_pre_pass_resume_cnt - startProf[i].apc_profiling.ape_pre_pass_resume_cnt << std::endl;
            std::cout << "ape_dshm_resume_cnt:" << endProf[i].apc_profiling.ape_dshm_resume_cnt - startProf[i].apc_profiling.ape_dshm_resume_cnt << std::endl;
            std::cout << "ape_resumed_type0_cnt:" << endProf[i].apc_profiling.ape_resumed_type0_cnt - startProf[i].apc_profiling.ape_resumed_type0_cnt << std::endl;
        }
        std::cout << std::dec;
    }

}

static void gpuUtilizationThread() {
    uint32_t uiBuffPos = 0;
    uint32_t uiApcMask = g_apc_mask;
    apcRunningData_t uaRunningCntsNow[32] = {0};
    apcRunningData_t uaRunningCntsLast[32] = {0};

    mapsStartRunningPerfCnt(0, uiApcMask);
    mapsReadRunningPerfCnt(0, uiApcMask, uaRunningCntsLast);
    auto startTime = std::chrono::high_resolution_clock::now();
    for (uint32_t iLoops = 0; iLoops < (uint32_t)g_test_loop; iLoops++) {
        std::cout << "<<************************test loop:" << std::dec << iLoops << "************************>>" << std::endl;
        gpuKernelRunThread();
    }
    mapsReadRunningPerfCnt(0, uiApcMask, uaRunningCntsNow);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "during time:" << std::dec << duration.count() << " ms" << std::endl;
    for (int i = 0; i < g_apc_num; i++) {
        std::cout << "APC" << std::dec << i << " running/periodic: " << std::hex << uaRunningCntsNow[i].runingCnt - uaRunningCntsLast[i].runingCnt;
        std::cout << "/" << uaRunningCntsNow[i].periodicCnt - uaRunningCntsLast[i].periodicCnt << std::endl;
    }
    std::cout << std::endl;
    startTime = endTime;
    memcpy(uaRunningCntsLast, uaRunningCntsNow, sizeof(apcRunningData_t) * 32);
    mapsStopRunningPerfCnt(0, uiApcMask);
}

static void startProfileTest()
{
    gpuUtilizationThread();
}


static int stringToUintArray(std::stringstream &ss_, uint32_t *uintArray, int ArraySize) {
    int i = 0;
    std::string token;

    memset(uintArray, 0, sizeof(uint32_t) * ArraySize);
    while (std::getline(ss_, token, ',')) {
        uintArray[i++] = std::stoull(token, nullptr, 0);
        if (i >= ArraySize)
            break;
    }
    return i;
}

#define HELP_TAB_SIZE 4
static void dumpHelp()
{
    std::cout << "TEST OPTIONS:" << std::endl;
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-t test index" << std::endl;
    for (int i = 0; i < ARRAY_SIZE(g_test_cmd_tab); i++) {
        std::cout << std::string(HELP_TAB_SIZE * 2, ' ') << std::dec << i << " [" << g_test_cmd_tab[i].name << "]" << std::endl;
        std::cout << std::string(HELP_TAB_SIZE * 3, ' ') << "- " << g_test_cmd_tab[i].helpStr << std::endl;
    }
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-N data number." << std::endl;
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-l test loop." << std::endl;
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-G grid cnt." << std::endl;
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-B block cnt." << std::endl;
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-m run kernel in apc mask." << std::endl;
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-k kernel ids profiling array." << std::endl;
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-g workgroup ids profiling array." << std::endl;
    std::cout << std::string(HELP_TAB_SIZE, ' ') << "-w warp ids profiling array." << std::endl;
}

static void dumpCmdParas()
{
    std::cout << "test loop:   " << std::dec << g_test_loop << std::endl;
    std::cout << "grid_cnt:    " << std::dec << g_grid_cnt << std::endl;
    std::cout << "block_cnt:   " << std::dec << g_block_cnt << std::endl;
    std::cout << "data numbers:0x" << std::hex << g_N << std::endl;
    std::cout << "test_index:  " << std::dec << g_test_index << std::endl;
    std::cout << "apc_mask:    0x" << std::hex << g_apc_mask << std::endl;
    std::cout << "apc_num:     " << std::dec << g_apc_num << std::endl;
    std::cout << "perf_kids:   " << std::dec << g_perf_kids[0] << "," << g_perf_kids[1] << "," << g_perf_kids[2] << "," << g_perf_kids[3] << std::endl;
    std::cout << "perf_wgids:  " << std::dec << g_perf_wgids[0] << "," << g_perf_wgids[1] << "," << g_perf_wgids[2] << "," << g_perf_wgids[3] << std::endl;
    std::cout << "perf_warpids:" << std::dec << g_perf_warpids[0] << "," << g_perf_warpids[1] << "," << g_perf_warpids[2] << "," << g_perf_warpids[3] << std::endl;
}

static int parseOptions(int argc, char *argv[]) {
    int opt;
    std::stringstream ss;

    while((opt = getopt(argc, argv, "l:G:B:t:m:k:g:w:N:h")) != -1) {
        switch(opt) {
            case 'l':
                g_test_loop = std::atoi(optarg);
            break;
            case 'G':
                g_grid_cnt = std::stoull(optarg, nullptr, 0);
            break;
            case 'B':
                g_block_cnt = std::stoull(optarg, nullptr, 0);
            break;
            case 't':
                g_test_index = std::atoi(optarg);
            break;
            case 'm':
                g_apc_mask = std::stoull(optarg, nullptr, 0);
                if (!g_apc_mask)
                    g_apc_mask = 0xFFFFFFFF;
                g_apc_num = 0;
                for(int i = 0; i < 32; i++) {
                    if (g_apc_mask & (1 << i))
                        g_apc_num++;
                }
            break;
            case 'N':
                g_N = std::stoull(optarg, nullptr, 0);
            break;
            case 'k':
                ss.str(optarg);
                ss.clear();
                stringToUintArray(ss, g_perf_kids, PROF_KERNEL_MAX);
            break;
            case 'g':
                ss.str(optarg);
                ss.clear();
                stringToUintArray(ss, g_perf_wgids, PROF_KERNEL_MAX);
            break;
            case 'w':
                ss.str(optarg);
                ss.clear();
                stringToUintArray(ss, g_perf_warpids, PROF_KERNEL_MAX);
            break;
            case '?':
                std::cout << "error opt: -" << static_cast<char>(optopt) << " !" << std::endl;
                [[fallthrough]];
            case 'h':
                dumpHelp();
                dumpCmdParas();
                return -1;
            break;
            default:
            break;
        }
    }
    dumpCmdParas();
    return 0;
}

int main(int argc, char *argv[]) {
    if (parseOptions(argc, argv) < 0)
        return 0;
    startProfileTest();
    return 0;
}

