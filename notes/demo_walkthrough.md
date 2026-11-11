# 官方 Demo 工程结构与运行原理解析 (Demo Walkthrough)

## 1. 概述与工程结构

在目标 LPU 架构上，SCALE 是一种针对异构计算的统一编程模型，语法风格与 CUDA 保持高度一致，但底层硬件模型有着根本性的创新：**SPU 与 MPU 硬件解耦** 以及 **显式管理的多 Bank 片上 DM（Data Memory）**。

官方 Demo 位于 `kernels/official_demo/`，包含以下核心文件：
- `main.sc`: Host 侧控制逻辑，包含 Host 内存分配、设备内存申请与物理地址获取、数据初始化与搬移、Kernel 启动 (`<<<>>>`)、结果回读与 CPU 精度校验。
- `kernels.sc`: SPU/MPU 设备侧核心实现，包含 `__global__` 入口、片上 DM 内存分配、DMA 异步搬运、MPU 计算调用以及同步机制。
- `mpu_add_u32.m.asm`: 纯 MPU 汇编实现示例（展示了通过 SPU 配置 MREG/KREG 寄存器后调用 MPU 汇编函数的传统底层流程）。
- `makefile`: 针对异构编译器（SCALE Clang/LLVM）的多阶段编译与链接流程。

---

## 2. 编译与链接流水线 (Compilation & Linking)

SCALE 编译器前端基于 Clang/LLVM，一次编译过程需要同时处理 Host (x86_64) 与 Device (LPU SPU/MPU) 两套体系结构。

### 2.1 头文件与路径分离
- **Host 包含路径** (`-Xarch_host -I...`)：
  - `/opt/maps/include`: MAPS 运行时系统 API (`maps_runtime_api.h` 等)。
  - `/usr/include/c++/11` 及 Linux 系统库。
  - `/opt/maps/toolchain/include/scale_include/llvm`: LLVM 基础支持库。
- **Device 包含路径** (`-Xarch_device -I...`)：
  - `/opt/maps/toolchain/include/scale_include/c++/v1`: 设备端轻量 C++ 标准库。
  - `/opt/maps/toolchain/include/scale_include/newlib`: 嵌入式 C 标准库。
  - `/opt/maps/toolchain/include/scale_include/$(SCALE_ARCH)-lib`: 目标设备硬件驱动与队列。
  - `-Xarch_device -nostdlibinc`: 严禁设备端引入 Host 侧的标准系统头文件。

### 2.2 编译阶段
```makefile
# 1. 编译 .sc 源码到 .o
$(TC_BIN)/clang++ kernels.sc -c [INCLUDES] -nostdlibinc $(FLAGS)

# 2. 编译 MPU 汇编 (.m.asm) 到 .o
$(TC_BIN)/clang++ -c -scale-mpu-asm mpu_add_u32.m.asm -o mpu_add_u32.o $(FLAGS)

# 3. 统一链接成最终 ELF 可执行文件
$(TC_BIN)/clang++ --scale-link -o a.out 2.o mpu_add_u32.o main.o kernels.o \
    -lscalehost -lmaps -lmapskmd -lllcdsc_kernel -lelf -lstdc++ -lm -lgcc -lc
```

`--scale-link` 会调用 `ld.lld`，将 Host 二进制与 Device 侧的内核固件包打包嵌入，在运行时由 `libscalehost.so` / `libmaps.so` 动态加载到仿真环境的协处理器与 APE 中。

---

## 3. Host 侧内存与 Kernel 生命周期

### 3.1 内存生命周期
目标 LPU 芯片目前在仿真及硬件架构上采用**物理地址直访模式**（Device SPU 的 DMA 引擎直接操作 DDR 物理地址）。因此 Host 侧代码有特定的内存管理模式：

```cpp
// 1. Host 内存申请
unsigned *h_A, *h_B, *h_C;
mapsHostAlloc((void**)&h_A, N * sizeof(unsigned), 0);
mapsHostAlloc((void**)&h_B, N * sizeof(unsigned), 0);
mapsHostAlloc((void**)&h_C, N * sizeof(unsigned), 0);

// 2. Device 虚拟地址申请
unsigned *d_A, *d_B, *d_C;
mapsMalloc((void**)&d_A, N * sizeof(unsigned));
mapsMalloc((void**)&d_B, N * sizeof(unsigned));
mapsMalloc((void**)&d_C, N * sizeof(unsigned));

// 3. 数据 Host -> Device 拷贝
mapsMemcpy(d_A, h_A, N * sizeof(unsigned), mapsMemcpyHostToDevice);
mapsMemcpy(d_B, h_B, N * sizeof(unsigned), mapsMemcpyHostToDevice);

// 4. 获取 DDR 物理地址 (PA)
// 这是目标架构必须步骤：传入 Kernel 参数的指针必须是 Device 的 64 位物理地址！
uint64_t d_A_pa, d_B_pa, d_C_pa;
mapsMemobjGetPhysicalAddress(d_A, &d_A_pa);
mapsMemobjGetPhysicalAddress(d_B, &d_B_pa);
mapsMemobjGetPhysicalAddress(d_C, &d_C_pa);
```

### 3.2 Kernel 启动与同步
```cpp
dim3 blockDim(16, 1, 1);
dim3 gridDim(4, 1, 1);

// 传入物理地址，强转为 __DDR 修饰的指针
vaddu32<<<gridDim, blockDim>>>((__DDR unsigned*)d_C_pa, 
                               (__DDR unsigned*)d_A_pa, 
                               (__DDR unsigned*)d_B_pa, 
                               N_);

// 等待设备完成
mapsDeviceSynchronize();

// 将结果回读至 Host 校验
mapsMemcpy(h_C, d_C, N * sizeof(unsigned), mapsMemcpyDeviceToHost);
```

---

## 4. Device 侧：SPU 与 MPU 的协同架构

### 4.1 APE 硬件角色分工
- **SPU (Scalar Processing Unit)**：
  - 乱序执行标量核，支持标准控制流、分支、地址计算。
  - 控制内部 DMA 引擎，发起 DDR <-> DM 的数据搬运。
  - 维护硬件 Barrier 和任务调度。
  - 组织数据并向 MPU 发起异步/同步计算任务。
- **MPU (Matrix/Vector Processing Unit)**：
  - VLIW 超长指令字向量引擎，内含 512 位宽向量寄存器。
  - **无权直接访问 DDR**，计算的所有源操作数和目的操作数必须位于片上 DM（Data Memory）。
  - 由 SPU 触发启动，完成后向 SPU 返回完成信号。

### 4.2 显式管理片上 DM (Memory Regions)
每个 APC 拥有 24 个 DM Bank。在 C 语言中通过属性修饰符显式划分：
- `__mem0__`, `__mem1__`, `__mem2__`: 线程私有/局部 DM 缓冲区，必须连续编号不可跳号。
- `__shared__`: APC 内部 4 个 APE 共享的 DM 区域。
- 建议单区域 `<= 32KB`，对齐到 64 字节（512 位向量对齐）。

### 4.3 核心执行流分析 (`vaddu32`)
```cpp
__global__ void vaddu32(__DDR unsigned *dst, __DDR unsigned *src1, __DDR unsigned *src2, unsigned N)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 1. 在片上 DM 分配对齐的向量缓存区
  __mem0__ unsigned local_src1[32] __attribute__((aligned(64)));
  __mem1__ unsigned local_src2[32] __attribute__((aligned(64)));
  __mem2__ unsigned local_dst[32]  __attribute__((aligned(64)));

  auto block = cooperative_groups::this_thread_block();
  using barrier_t = scale::barrier<scale::thread_scope_block>;
  __shared__ barrier_t bar;
  if (threadIdx.x == 0) init(&bar, blockDim.x * blockDim.y * blockDim.z);
  __syncthreads();

  // 2. SPU 驱动 DMA 引擎异步将数据从 DDR 搬入 DM
  memcpy_async(block, local_src1, (src1 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);
  memcpy_async(block, local_src2, (src2 + tid * N), scale::aligned_size_t<4>(N*sizeof(int)), bar);

  // 3. 等待 DMA 搬运完毕
  bar.arrive_and_wait();

  // 4. SPU 调用 MPU 进行向量计算 (C 风格向量代码)
  mpu_vadd_v16u32((v16u32 *)local_dst, (v16u32 *)local_src1, (v16u32 *)local_src2, N/16);
  MPU_SYNC(); // 等待 MPU 执行完成

  // 5. SPU 驱动 DMA 引擎将结果异步写回 DDR
  memcpy_async(block, &dst[tid*N], (unsigned *)local_dst, scale::aligned_size_t<4>(N*sizeof(int)), bar);
  bar.arrive_and_wait();
}
```

### 4.4 MPU C vs. MPU 汇编
在官方 demo 中：
- `mpu_add_u32.m.asm` 展示了纯汇编写法，需要手动配置 MREG 寄存器参数并跳转返回。
- `mpu_vadd_v16u32` 展示了 **MPU C (`KERNEL_M`)** 写法：
```cpp
__device__ KERNEL_M void mpu_vadd_v16u32(v16u32 *dst, v16u32 *src1, v16u32 *src2, unsigned N)
{
  for (unsigned i = 0; i < N; i++) {
      dst[i] = src1[i] + src2[i];
  }
}
```
编译器会自动将 `KERNEL_M` 函数编译为针对 MPU VLIW 流水线的高效向量机器码。**在后续 LLM 算子实现中，我们将优先使用 MPU C++ 配合向量内置类型与 Intrinsics**，既具备高级语言的生产力与类型安全，又能直接发挥 512 位向量流水线的算力。
