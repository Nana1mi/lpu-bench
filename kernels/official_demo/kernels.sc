/**
 * kernel_vadd.s.c
 *
 * Created on: 2024-10-03 13:07:45
 *      Author: yangk
 */

#include "scale.h"
#include "maps_runtime_api.h"
#include "scale/barrier"
#include "cooperative_groups.h"
#define size_t unsigned long
#include "aurora_s.h"
#include "aurora.h"
#include <aurora_s-intrin.h>
#include "maops.h"
#if USE_LLCDSC
#include "aurora.h"
#include "dma.h"
#endif

// UNIT_TEST global data
__device__ __attribute__((aligned(64))) unsigned global_src1[] = {
#include "input.data"
};
__device__ __attribute__((aligned(64))) unsigned global_src2[] = {
#include "input.data"
#include "input.data"
#include "input.data"
};
__constant__ __attribute__((aligned(64))) unsigned global_dst1[sizeof(global_src1)/sizeof(global_src1[0])];
__constant__ __attribute__((aligned(64))) unsigned global_dst2[sizeof(global_src2)/sizeof(global_src2[0])];

__global__ __system_cache__ void printf_debug(__DDR unsigned *dst, __DDR unsigned *src) {
  uint8_t ape_id = __aurora_s_get_core_id() & 0x3;
  uint8_t apc_id = __aurora_s_get_core_id() >> 2;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("outer: ===tid = %d, apc_id = %d, ape_id = %d ===\n", tid, apc_id, ape_id);
  if ((apc_id == 0) && (ape_id == 0)) {
    printf("inner: ===tid = %d, apc_id = %d, ape_id = %d ===\n", tid, apc_id, ape_id);
    dst[tid] = src[tid];
  }
  return;
}

__global__ void profiler_devbuf() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  scale_init_maps_profiler_buffer_addr();
  printf("__scale_profiler_buffer_addr is 0x%llx\n", __scale_profiler_buffer_addr);
  __DDR unsigned *ptr = (__DDR unsigned*)(__scale_profiler_buffer_addr);
  if (!ptr)
    printf("profiler_devbuf <%d>: ptr is null\n", tid);
  else {
    *ptr = 0x11; // non zero
    printf("profiler_devbuf <%d>: ptr is 0x%llx, val is 0x%x\n", tid, (uint64_t) __scale_profiler_buffer_addr, *ptr);
  }
}

__global__ void print_stress() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 70; i++)
      printf("[%d]<======================>", i);
    printf(":4:ammu.c                   :589 : 0449399394d us: [pid:5227 tid: 7f72ec400840] unmap success!!!!\n:4:device.cpp               :536 : 0449399553d us: [pid:5227 tid: 7f72ec400840] releaseQueue base_address: 0x7e5aea96d300 refCount: (0)\n:4:device.cpp               :548 : 0449399561d us: [pid:5227 tid: 7f72ec400840] Deleting hardware queue 0x7e5aea96d300 with refCount 0\n:4:ccb.c                    :322 : 0449403130d us: [pid:5227 tid: 7f72ec400840] ccb_kccb_emit: wptr:42 rptr:41\n:4:maps_stream.cpp          :417 : 0449437178d us: [pid:5227 tid: 7f72ec400840] mapsStreamDestroy: Returned mapsSuccess : \n:4:maps_event.cpp           :390 : 0449437212d us: [pid:5227 tid: 7f72ec400840] [32m mapsEventDestroy ( event:0x55586226d500 ) [0m\n:4:maps_event.cpp           :328 : 0449437399d us: [pid:5227 tid: 7f72ec400840] imapsEventDestroy: Returned mapsSuccess : \n:4:maps_event.cpp           :391 : 0449437408d us: [pid:5227 tid: 7f72ec400840] mapsEventDestroy: Returned mapsSuccess : : duration: 196d us\n:4:maps_stream.cpp          :368 : 0449437417d us: [pid:5227 tid: 7f72ec400840] [32m mapsStreamDestroy ( stream:0x55586226d5c0 ) [0m\n:4:amd_kfd_driver.cpp       :194 : 0449437440d us: [pid:5227 tid: 7f72ec400840] KfdDriver::FreeMemory block mem:0x55586226d430 size:18874368 offset:0 mem_bo:0x55586226dc00 seg:3 flags:0x21 gpu_va:0x7e5c09200000 gpu_pa:0x509e00000 cpu_pa:0x1109e00000\n:4:ammu.c                   :589 : 0449439292d us: [pid:5227 tid: 7f72ec400840] unmap success!!!!\n:4:device.cpp               :536 : 0449439475d us: [pid:5227 tid: 7f72ec400840] releaseQueue base_address: 0x7e5aea961000 refCount: (0)\n:4:device.cpp               :548 : 0449439485d us: [pid:5227 tid: 7f72ec400840] Deleting hardware queue 0x7e5aea961000 with refCount 0\n:4:ccb.c                    :322 : 0449443240d us: [pid:5227 tid: 7f72ec400840] ccb_kccb_emit: wptr:43 rptr:42\n:4:maps_stream.cpp          :417 : 0449478755d us: [pid:5227 tid: 7f72ec400840] mapsStreamDestroy: Returned mapsSuccess : \n[2025-11-26 18:36:29.526][5227][info][Utils.cpp(536): scaleGetKernelName][scalehost] deviceName: _Z11clean_cacheiiii\n[2025-11-26 18:36:29.526][5227][info][Utils.cpp(613): scaleGetKernelTextDataSize][scalehost]     ld.lld --relocatable -o /var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/build_tc/_Z11clean_cacheiiii.out  --just-symbols=/var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/../toolchain/lib/scale/release//scale_scheduler_apc_strip.out --allow-multiple-definition /var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/extract_tc/*.o /var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/../toolchain/lib/scale/release//aurora_dlmt.o /var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/../toolchain/lib/scale/release//aurora_sync.o /var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/../toolchain/lib/scale/release//device_builtin_vars.o /var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/../toolchain/lib/scale/release//aurora_callm_queue_args_buff.s.o  -L/var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/../toolchain/lib/scale/release/ -lm -lgloss -lclang_rt.builtins-aurora_s -laurora -lc++ -lc++abi -lscale -lc   -T/var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/../toolchain/lib/scale/release//apc-ai.ld --gc-sections --entry=_Z11clean_cacheiiii\n[2025-11-26 18:36:29.548][5227][info][Utils.cpp(689): scaleLinkAndRecordSize][scalehost]     ld.lld -o /var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/build_tc/_Z11clean_cacheiiii.out  --allow-multiple-definition  -T/var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/../toolchain/lib/scale/release//apc-ai.ld --gc-sections --entry=_Z11clean_cacheiiii  --defsym=__scale_spu_text_base=0x400300000 --defsym=MATH_TABLE_ADDR=0x7e5c00000000 --defsym=__scale_profiler_buffer_addr=0x0 /var/jenkins/workspace/ai-mr_maps-umd_MR-306/maps-umd/build_tc/_Z11clean_cacheiiii.out\n");
    for (int i = 0; i < 70; i++)
      printf("[%d]<======================>", i);
  }
}

__global__ void print_another() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("print_another <%d>%lld: ", tid, 2ULL);
  float e_f = 2.72f;
  double e_d = 2.718281828459;
  printf("warn float_e:%f, double_e:%f, exponent_e:%e\n", e_f, e_d, e_d);
}

__global__ void print1() {
  __mem0__ __fix0__ int a;
  a = blockIdx.x * blockDim.x + threadIdx.x;
  printf("printf1 on %d\n", a);
}

__global__ void print() {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("print [%d]%lld: ", tid, 2ULL);
  float pi_f = 3.14f;
  double pi_d = 3.14159265357;
  printf("warn float_pi:%f, double_pi:%f, exponent_pi:%e\n", pi_f, pi_d, pi_d);
#if 0
  const char *str = "abcd";
  printf("m%s%s\n", str, "xyz");
  const char *str2 = "abcd123456111111111111111111111111111111111111111111111111111"
        "1111111111111111111111111111111111111111111111111111111111111111111"
        "11111111111111111111111111111111111111111111111111111111111"
        "1111111111111111111111111111111111111111111111111111111111111111"
        "111111111111111111111111111111111111111111111111111111111111111111111"
        "11111111111111111111111111111111111111111111111111111111111111111111111"
        "111111111111111111111111111111111111111111111111111111111111111111111111111"
        "11111111111111111111111111111111111111111111111111111111111111111111111111111"
        "111111111111111111111111111111111111111111111111111111111111111111111111111111111";
  printf("%d%s%s\n", 100, str2, "xyz");
  int a = 100;
  printf("%d%d%d%d %d%d%d%d %d%d%d%d %d%d%d%d %d%d%d%d %d%d%d%d %d%d%d%d %d%d%d%d\n",
          a,a,a,a, a,a,a,a, a,a,a,a, a,a,a,a,
          a,a,a,a, a,a,a,a, a,a,a,a, a,a,a,a);
#endif
}

__global__ void simple_copy32(__DDR unsigned *dst, __DDR unsigned *src) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  dst[tid] = src[tid];
  return;
}

__global__ void simple_copy32_stress(__DDR unsigned *dst, __DDR unsigned *src) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < 32 * 16; i++) {
    dst[tid] = src[tid];
  }
  return;
}

__global__ void test_counter(__DDR unsigned *dst, unsigned num)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) {
        (*(dst + tid))++;
    }

    return;
}
// ---------------------------------------------------------------------------//
// kernel vaddu32 N mode 8 threads
__device__ KERNEL_M void mpu_vadd_v16u32(v16u32 *dst, v16u32 *src1, v16u32 *src2, unsigned N)
{
  for (unsigned i = 0; i < N; i++) {
      dst[i] = src1[i] + src2[i];
  }
}

//__device__ unsigned trace0[16];
//__device__ unsigned trace1[16];
//__device__ unsigned long long trace2[16];
//__device__ unsigned trace3[16];

__global__ void vaddu32_nomemcpy(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __mem0__ unsigned local_src1[32] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[32] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[32] __attribute__((aligned(64)));
  
  for (int i = 0; i < N; i++) {
    local_src1[i] = src1[tid * N + i];
    local_src2[i] = src2[tid * N + i];
  }

  mpu_vadd_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N / 16);
  MPU_SYNC();

  for (int i = 0; i < N; i++) {
    dst[tid * N + i] = local_dst[i];
  }
  return;
}

#if USE_LLCDSC
__global__ void vaddu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N)
{
  DMAHandle handle, handle_out;
  DmaFetchData fetch_data_in, fetch_data_out;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int asid = ((scale_warp_info_t *)curWarpInfo)->warp_info.kernel_info->as_id;

  __mem0__ unsigned local_src1[32] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[32] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[32] __attribute__((aligned(64)));

//  trace2[tid] = ((unsigned long long)curWarpInfo);
//  trace3[tid] = ((unsigned *)curWarpInfo)[1];
//  trace0[tid] = tid + 100;
//  __syncthreads();
//  trace1[tid] = tid + 200;

  handle.set_cmd_tag(warpEntryIdx);
  handle.set_hbm_write_flag(0);
  handle.set_need_switch_flag(1);
  handle.set_user_asid(asid);
  handle_out.set_user_asid(asid);
  handle_out.set_cmd_tag(warpEntryIdx);
  handle_out.set_hbm_write_flag(1);
  handle_out.set_need_switch_flag(1);

  deviceMemcpyAsync(DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_src1), sizeof(int), N, N, 0, 0, 0),
                    DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(src1 + tid * N), sizeof(int), N, N, 0, 0, 0),
                    deviceMemcpyHbmToDm,
                    handle, fetch_data_in);
  wait(handle);
  deviceMemcpyAsync(DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_src2), sizeof(int), N, N, 0, 0, 0),
                    DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(src2 + tid * N), sizeof(int), N, N, 0, 0, 0),
                    deviceMemcpyHbmToDm,
                    handle, fetch_data_in);
  wait(handle);

  mpu_vadd_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N/16);
  MPU_SYNC();

  deviceMemcpyAsync(DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(dst + tid * N), sizeof(int), N, N, 0, 0, 0),
                  DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_dst), sizeof(int), N, N, 0, 0, 0),
                  deviceMemcpyDmToHbm,
                  handle_out, fetch_data_out);
  wait(handle_out);

  return;
}
#else
__global__ void vaddu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N)
{

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __mem0__ unsigned local_src1[32] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[32] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[32] __attribute__((aligned(64)));

  auto block = cooperative_groups::this_thread_block();
  // define shared barrier & count
  using barrier_t = scale::barrier<scale::thread_scope_block>;
  __shared__ barrier_t bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x * blockDim.y * blockDim.z);
  }
//  trace2[tid] = ((unsigned long long)curWarpInfo);
//  trace3[tid] = ((unsigned *)curWarpInfo)[1];
//  trace0[tid] = tid + 100;
  __syncthreads();
//  trace1[tid] = tid + 200;


  memcpy_async(block, local_src1, (src1 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);
  memcpy_async(block, local_src2, (src2 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);

  bar.arrive_and_wait();
  mpu_vadd_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N/16);
  MPU_SYNC();

  bar.arrive_and_wait();
  memcpy_async(block, &dst[tid*N], (unsigned *)local_dst, scale::aligned_size_t<4>(N*sizeof(int)), bar);

  bar.arrive_and_wait();
  return;
}
#endif

extern "C" __device__ KERNEL_M void mpu_add_u32(unsigned int length);

__device__ KERNEL_S void mpu_vadd_v16u32_spu(unsigned int input1_addr, unsigned int input2_addr, unsigned int output_addr, unsigned int length){
       //Below is byte unit

    v16u32 params_input1 = {
        input1_addr,  //KB0, 0 - 31
        input1_addr,  //KB1, 32 - 63
        input1_addr,  //KB2, 64 - 95
        input1_addr,  //KB3, 96- 127
        0x0000010<<2,    //KS0 -- 64 bytes, with 512bit, 128- 159
        0x100<<2,      //KS1 -- 1024 bytes, 160 - 191
        0x0,        //KS2, 192 - 223,
        0x0,        //KS3, 224 - 255
        0x00100010,      //KI0&KC0 -- KI0 = 16, KC0 = 16, 256 - 287
        0x01000100,      //KI1&KC1 -- KI1 = 256, KC1=256, 288 - 319
        0x0,      //KI2&KC2 -- KI2 = 0, KC2=0, 320 - 351
        0xffffffff,      //reserved, ignored bit, param is not need to fill, 352 - 383,
          0x00040000,         //KBnum-262144 with 256KB aka the whole DM bytes, 384 - 415
          0x0,                //ignored, reserved bit, param is not need to fill, 416 - 447
        0x00000f0c,         // Gran:0, CGran=bit 1, means CGran=4,KSize = 0, WarpSize=15, WarpSize^15=32768, with 32 word 1 bank, read 1bank at a time , read 16 word
        // 448 - 479
        0xffff0000          // KM all is bit 1, 480 - 511
        };

    v16u32 params_input2 = {
        input2_addr,  //KB0
        input2_addr,  //KB1
        input2_addr,  //KB2
        input2_addr,  //KB3 // 3
        0x0000010<<2,    //KS0 -- 64 bytes, with 512bit
        0x100<<2,      //KS1 -- 1024 bytes
        0x0,        //KS2
        0x0,        //KS3
        0x00100010,      //KI0&KC0 -- KI0 = 16, KC0 = 16
        0x01000100,      //KI1&KC1 -- KI1 = 256, KC1=256
        0x0,      //KI2&KC2 -- KI2 = 0, KC2=0
        0xffffffff,      //reserved, ignored bit
        0x00040000,         //KBnum-262144 with 256KB aka the whole DM bytes
        0x0,                //ignored, reserved bit
        0x00000f0c,         // Gran:0, CGran=bit 1, means CGran=4,KSize = 0, WarpSize=15, WarpSize^15=32768, with 32 word 1 bank, read 1bank at a time , read 16 word
        0xffff0000          // KM all is bit 1
        };

    v16u32 params_output = {
        output_addr,  //KB0
        output_addr,  //KB1
        output_addr,  //KB2
        output_addr,  //KB3 // 3
        0x0000010<<2,    //KS0
        0x100<<2,      //KS1
        0x0,        //KS2
        0x0,        //KS3
        0x00100010,      //KI0&KC0
        0x01000100,      //KI1&KC1
        0x0,            //KI2&KC2
        0xffffffff,      //
          length << 2,        // length is fp32 gran, here is byte gran
          0x0,
        0x00000f0c,        // same as input
        0xffff0000
        };

    __aurora_s_setMReg_var(0,*(v16u32*)&params_input1);
    __aurora_s_setMReg_var(1,*(v16u32*)&params_input2);
    __aurora_s_setMReg_var(2,*(v16u32*)&params_output);

    mpu_add_u32((length+15)/16);
    __aurora_s_getStatB();

    __aurora_s_fenceI();
    return;

}

__global__ void vaddu32_asm(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N)
{

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __mem0__ unsigned local_src1[32] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[32] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[32] __attribute__((aligned(64)));

  auto block = cooperative_groups::this_thread_block();
  // define shared barrier & count
  using barrier_t = scale::barrier<scale::thread_scope_block>;
  __shared__ barrier_t bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x * blockDim.y * blockDim.z);
  }
//  trace2[tid] = ((unsigned long long)curWarpInfo);
//  trace3[tid] = ((unsigned *)curWarpInfo)[1];
//  trace0[tid] = tid + 100;
  __syncthreads();
//  trace1[tid] = tid + 200;


  memcpy_async(block, local_src1, (src1 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);
  memcpy_async(block, local_src2, (src2 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);

  bar.arrive_and_wait();
  mpu_vadd_v16u32_spu(static_cast<unsigned int>((uintptr_t)local_dst),
                        static_cast<unsigned int>((uintptr_t)local_src1),
                        static_cast<unsigned int>((uintptr_t)local_src2),
                        N/16);

  bar.arrive_and_wait();
  memcpy_async(block, &dst[tid*N], (unsigned *)local_dst, scale::aligned_size_t<4>(N*sizeof(int)), bar);

  bar.arrive_and_wait();
  return;
}


// ---------------------------------------------------------------------------//
// kernel vmulu32 C mode
__device__ KERNEL_M void mpu_vmul_v16u32(v16u32 *dst, v16u32 *src1, v16u32 *src2, unsigned N)
{
  for (unsigned i = 0; i < N; i++) {
      dst[i] = src1[i] * src2[i];
  }
}

#if USE_LLCDSC
__global__ void vmulu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N, __DDR unsigned long long *addr)
{
  DMAHandle handle, handle_out;
  DmaFetchData fetch_data_in, fetch_data_out;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int asid = ((scale_warp_info_t *)curWarpInfo)->warp_info.kernel_info->as_id;

  __mem0__ unsigned local_src1[64] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[64] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[64] __attribute__((aligned(64)));

  addr[0] = (unsigned long long)addressof(APE0, &local_src1);
  addr[1] = (unsigned long long)addressof(APE1, &local_src1);
  addr[2] = (unsigned long long)addressof(APE2, &local_src1);
  addr[3] = (unsigned long long)addressof(APE3, &local_src1);
  addr[4] = (unsigned long long)addressof(APE0, &local_src2);
  addr[5] = (unsigned long long)addressof(APE1, &local_src2);
  addr[6] = (unsigned long long)addressof(APE2, &local_src2);
  addr[7] = (unsigned long long)addressof(APE3, &local_src2);
  addr[8] = (unsigned long long)addressof(APE0, &local_dst);
  addr[9] = (unsigned long long)addressof(APE1, &local_dst);
  addr[10] = (unsigned long long)addressof(APE2, &local_dst);
  addr[11] = (unsigned long long)addressof(APE3, &local_dst);

  handle.set_cmd_tag(warpEntryIdx);
  handle.set_hbm_write_flag(0);
  handle.set_need_switch_flag(0);
  handle.set_user_asid(asid);
  handle_out.set_user_asid(asid);
  handle_out.set_cmd_tag(warpEntryIdx);
  handle_out.set_hbm_write_flag(1);
  handle_out.set_need_switch_flag(0);

  deviceMemcpyAsync(DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_src1), sizeof(int), N, N, 0, 0, 0),
                    DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(src1 + tid * N), sizeof(int), N, N, 0, 0, 0),
                    deviceMemcpyHbmToDm,
                    handle, fetch_data_in);
  wait(handle);
  deviceMemcpyAsync(DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_src2), sizeof(int), N, N, 0, 0, 0),
                    DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(src2 + tid * N), sizeof(int), N, N, 0, 0, 0),
                    deviceMemcpyHbmToDm,
                    handle, fetch_data_in);
  wait(handle);

  mpu_vmul_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N/16);
  MPU_SYNC();

  deviceMemcpyAsync(DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(dst + tid * N), sizeof(int), N, N, 0, 0, 0),
                  DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_dst), sizeof(int), N, N, 0, 0, 0),
                  deviceMemcpyDmToHbm,
                  handle_out, fetch_data_out);
  wait(handle_out);

  return;
}
#else
__global__ void vmulu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N, __DDR unsigned long long *addr)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;


  __mem0__ unsigned local_src1[64] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[64] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[64] __attribute__((aligned(64)));

  addr[0] = (unsigned long long)addressof(APE0, &local_src1);
  addr[1] = (unsigned long long)addressof(APE1, &local_src1);
  addr[2] = (unsigned long long)addressof(APE2, &local_src1);
  addr[3] = (unsigned long long)addressof(APE3, &local_src1);
  addr[4] = (unsigned long long)addressof(APE0, &local_src2);
  addr[5] = (unsigned long long)addressof(APE1, &local_src2);
  addr[6] = (unsigned long long)addressof(APE2, &local_src2);
  addr[7] = (unsigned long long)addressof(APE3, &local_src2);
  addr[8] = (unsigned long long)addressof(APE0, &local_dst);
  addr[9] = (unsigned long long)addressof(APE1, &local_dst);
  addr[10] = (unsigned long long)addressof(APE2, &local_dst);
  addr[11] = (unsigned long long)addressof(APE3, &local_dst);

  auto block = cooperative_groups::this_thread_block();
  // define shared barrier & count
  using barrier_t = scale::barrier<scale::thread_scope_block, scale::__empty_completion, false/* do not switch */>;
  __shared__ barrier_t bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x * blockDim.y * blockDim.z);
  }
  __syncthreads();

  memcpy_async(block, local_src1, (src1 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);
  memcpy_async(block, local_src2, (src2 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);

  bar.arrive_and_wait();
  mpu_vmul_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N/16);
  MPU_SYNC();

  bar.arrive_and_wait();
  memcpy_async(block, &dst[tid*N], (unsigned *)local_dst, scale::aligned_size_t<4>(N*sizeof(int)), bar);

  bar.arrive_and_wait();
  return;
}
#endif
// kernel vmulu32 end
// ---------------------------------------------------------------------------//


// ---------------------------------------------------------------------------//
// kernel vsubu32
#if 1
__device__ KERNEL_M void mpu_vsub_v16u32(v16u32 *dst, v16u32 *src1, v16u32 *src2, unsigned N)
{
  for (unsigned i = 0; i < N; i++) {
      dst[i] = src1[i] - src2[i];
  }
}
#endif

__global__ void vsubu32_nomemcpy(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __mem0__ unsigned local_src1[32] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[32] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[32] __attribute__((aligned(64)));
  
  for (int i = 0; i < N; i++) {
    local_src1[i] = src1[tid * N + i];
    local_src2[i] = src2[tid * N + i];
  }

  mpu_vsub_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N / 16);
  MPU_SYNC();

  for (int i = 0; i < N; i++) {
    dst[tid * N + i] = local_dst[i];
  }
  return;
}

#if USE_LLCDSC
__global__ void vsubu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N)
{
  DMAHandle handle, handle_out;
  DmaFetchData fetch_data_in, fetch_data_out;
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int asid = ((scale_warp_info_t *)curWarpInfo)->warp_info.kernel_info->as_id;

  __mem0__ unsigned local_src1[64] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[64] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[64] __attribute__((aligned(64)));

  handle.set_cmd_tag(warpEntryIdx);
  handle.set_hbm_write_flag(0);
  handle.set_need_switch_flag(1);
  handle.set_user_asid(asid);
  handle_out.set_user_asid(asid);
  handle_out.set_cmd_tag(warpEntryIdx);
  handle_out.set_hbm_write_flag(1);
  handle_out.set_need_switch_flag(1);

  deviceMemcpyAsync(DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_src1), sizeof(int), N, N, 0, 0, 0),
                    DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(src1 + tid * N), sizeof(int), N, N, 0, 0, 0),
                    deviceMemcpyHbmToDm,
                    handle, fetch_data_in);
  wait(handle);
  deviceMemcpyAsync(DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_src2), sizeof(int), N, N, 0, 0, 0),
                    DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(src2 + tid * N), sizeof(int), N, N, 0, 0, 0),
                    deviceMemcpyHbmToDm,
                    handle, fetch_data_in);
  wait(handle);

  mpu_vsub_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N/16);
  MPU_SYNC();

  deviceMemcpyAsync(DataParam::construct_hbm_param(reinterpret_cast<uint64_t>(dst + tid * N), sizeof(int), N, N, 0, 0, 0),
                  DataParam::construct_dm_param(reinterpret_cast<uint32_t>(local_dst), sizeof(int), N, N, 0, 0, 0),
                  deviceMemcpyDmToHbm,
                  handle_out, fetch_data_out);
  wait(handle_out);

  return;
}
#else
__global__ void vsubu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __mem0__ unsigned local_src1[64] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[64] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[64] __attribute__((aligned(64)));

  auto block = cooperative_groups::this_thread_block();
  // define shared barrier & count
  using barrier_t = scale::barrier<scale::thread_scope_block>;
  __shared__ barrier_t bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x * blockDim.y * blockDim.z);
  }
  __syncthreads();

  memcpy_async(block, local_src1, (src1 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);
  memcpy_async(block, local_src2, (src2 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);

  bar.arrive_and_wait();
  mpu_vsub_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N/16);
  MPU_SYNC();

  bar.arrive_and_wait();
  memcpy_async(block, &dst[tid*N], (unsigned *)local_dst, scale::aligned_size_t<4>(N*sizeof(int)), bar);

  bar.arrive_and_wait();
  return;
}
#endif
// kernel vsubu32 end
// ---------------------------------------------------------------------------//

extern "C" __device__ KERNEL_M void mpu_tanh_fp32(unsigned int length);

__device__ KERNEL_S int spu_tanh_fp32(unsigned int input_addr, unsigned int table_0, unsigned int table_1, unsigned int output_addr, unsigned int length) {
    v16u32 params_input = {
        input_addr,	    // KB0, 0 - 31
        input_addr,	    // KB1, 32 - 63
        input_addr,	    // KB2, 64 - 95
        input_addr,	    // KB3, 96- 127
        0x0000010<<2,   // KS0 -- 64 bytes, with 512bit, 128 - 159
        0x100<<2,		// KS1 -- 1024 bytes, 160 - 191
        0x0,			// KS2, 192 - 223,
        0x0,			// KS3, 224 - 255
        0x00100010,		// KI0&KC0 -- KI0 = 16, KC0 = 16, 256 - 287
        0x01000100,		// KI1&KC1 -- KI1 = 256, KC1=256, 288 - 319
        0x0,			// KI2&KC2 -- KI2 = 0, KC2=0, 320 - 351
        0xffffffff,		// reserved, ignored bit, param is not need to fill, 352 - 383
        0x00040000,     // KBnum-262144 with 256KB aka the whole DM bytes, 384 - 415
        0x0,            // ignored, reserved bit, param is not need to fill, 416 - 447
        0x00000f0c,     // Gran:4, CGran=bit 1, means CGran=4,KSize = 0, WarpSize=15, WarpSize^15=32768, with 32 word 1 bank, read 1 bank at a time , read 16 word, 448 - 479
        0xffff0000      // KM all is bit 1, 480 - 511
    };

    v16u32 params_table0 = {
		table_0 - 0x80,	// KB0
		table_0 - 0x80,	// KB1
		table_0 - 0x80,	// KB2
		table_0 - 0x80,	// KB3
		0x0,	        // KS0
		0x0,		    // KS1
		0x0,			// KS2
		0x0,			// KS3
		0x0,		    // KI0&KC0
		0x0,		    // KI1&KC1
		0x0,			// KI2&KC2
		0xffffffff,		// reserved, ignored bit
		0x00180000,     // KBnum
		0x0,            // ignored, reserved bit
        0x010c0f0c,     // offsetAddrmode=3, LSType=2
		0xffff0003      // index=3
	};

    v16u32 params_table1 = {
		table_1 - 0x80,	// KB0
		table_1 - 0x80,	// KB1
		table_1 - 0x80,	// KB2
		table_1 - 0x80,	// KB3
		0x0,	        // KS0
		0x0,		    // KS1
		0x0,			// KS2
		0x0,			// KS3
		0x0,		    // KI0&KC0
		0x0,		    // KI1&KC1
		0x0,			// KI2&KC2
		0xffffffff,		// reserved, ignored bit
		0x00180000,     // KBnum
		0x0,            // ignored, reserved bit
        0x010c0f0c,     // offsetAddrmode=3, LSType=2
		0xffff0003      // index=3
	};

	v16u32 params_output = {
		output_addr,	// KB0
		output_addr,	// KB1
		output_addr,	// KB2
		output_addr,	// KB3
		0x0000010<<2,	// KS0
		0x100<<2,		// KS1
		0x0,			// KS2
		0x0,			// KS3
		0x00100010,		// KI0&KC0
		0x01000100,		// KI1&KC1
		0x0,			// KI2&KC2
		0xffffffff,		//
		length << 2,	// length is fp32 gran, here is byte gran
		0x0,
		0x00000f0c,     // same as input
		0xffff0000
	};

    __aurora_s_setMReg_var(0,*(v16u32*)&params_input);
    __aurora_s_setMReg_var(1,*(v16u32*)&params_table0);
    __aurora_s_setMReg_var(2,*(v16u32*)&params_table1);
    __aurora_s_setMReg_var(64,*(v16u32*)&params_output);

    //  0x80000000, 0x7fffffff, boundary  , upper bound
    //  a0        , a1        , a2        , a3
    //  a4        ,-start/step, 1/step    , start
    //  step     
    __aurora_s_setMReg_var(3,(v16u32){
        0x80000000, 0x7fffffff, 0x3f19999a, 0x41077800,
        0x00000000, 0xbeaaa9b9, 0x3e0855ab, 0xbd561d71,
        0x3c80e124, 0xc1800000, 0x42000000, 0x3f000000,
        0x3d000000, 0x00000000, 0x00000000, 0x00000000,
    });

    unsigned int loop = (length+15) >> 4;
    mpu_tanh_fp32(loop);

    __aurora_s_getStatB();
	  __aurora_s_fenceI();

    return 0;
}

__global__ void vtanhfp32(__DDR float* dst, const __DDR float* src,
                          const unsigned int N_per_thread) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __mem0__ float local_src[TANH_TABLE_0_D_COUNT] __attribute__((aligned(64)));
  __mem1__ float local_dst[TANH_TABLE_0_D_COUNT] __attribute__((aligned(64)));
  __shared__ float s_tanh_table_0[TANH_TABLE_0_D_COUNT];
  __shared__ float s_tanh_table_1[TANH_TABLE_1_D_COUNT];
  unsigned long long tanh_table_0_addr = (unsigned long long)TANH_TABLE_0_D_ADDR;
  unsigned long long tanh_table_1_addr = (unsigned long long)TANH_TABLE_1_D_ADDR;

  auto block = cooperative_groups::this_thread_block();
  // define shared barrier & count
  using barrier_t = scale::barrier<scale::thread_scope_block>;
  __shared__ barrier_t bar;

  if (threadIdx.x == 0) {
    init(&bar, blockDim.x * blockDim.y * blockDim.z);
  }
  __syncthreads();

  memcpy_async(block, local_src, (src + tid * N_per_thread), scale::aligned_size_t<4>(N_per_thread*sizeof(float)), bar);
  memcpy_async(block, s_tanh_table_0, (__DDR float*)(tanh_table_0_addr), TANH_TABLE_0_D_SIZE, bar);
  memcpy_async(block, s_tanh_table_1, (__DDR float*)(tanh_table_1_addr), TANH_TABLE_1_D_SIZE, bar);

  bar.arrive_and_wait();

  spu_tanh_fp32(static_cast<unsigned int>((uintptr_t)local_src),
                         static_cast<unsigned int>((uintptr_t)s_tanh_table_0),
                         static_cast<unsigned int>((uintptr_t)s_tanh_table_1),
                         static_cast<unsigned int>((uintptr_t)local_dst), N_per_thread);
  bar.arrive_and_wait();

  memcpy_async(block, dst + tid * N_per_thread, (unsigned *)local_dst, scale::aligned_size_t<4>(N_per_thread*sizeof(float)), bar);

  bar.arrive_and_wait();
  return;
}
