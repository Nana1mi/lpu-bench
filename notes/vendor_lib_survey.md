# 厂商官方算子库调研与实测分析报告 (notes/vendor_lib_survey.md)

**调研对象**: 厂商提供的官方演示工程中的矩阵乘库 (`mablaslt`) 与算子库 (`maops`)  
**相关任务**: T5 (mablaslt 调研与运行验证), T6 (maops 与 FlashAttention 实现剖析)  
**实验环境**: WSL2 (Ubuntu 22.04) 宿主机构建，QEMU CModel (Linux 5.15, SCALE 驱动与运行时) 客体机仿真运行  
**度量基准**: 硬件仿真时钟周期 (`device_cycles` / `tick()`)，排除宿主机墙钟 (`sim_cycles` / `sim_ms`)  

---

## 1. 概述与背景

厂商近期提供了包含 37 个模块、1022 个文件的官方演示代码包，其中包含了核心的基础线性代数矩阵乘库 `mablaslt` 以及面向大模型算子的计算库 `maops`。这些官方库的内部实现直接反映了厂商在目标 LPU 芯片上的推荐软件架构、底层优化路径以及硬件特性的调用范式。

本项目针对上述官方库开展了系统性的源码调研、依赖解析、构建测试以及运行时验证，重点评估其对硬件拓扑的依赖性、算子实现技术路线（C Intrinsic vs 手工微码汇编）以及在 CModel 仿真环境中的可测性。

---

## 2. mablaslt 矩阵乘库调研分析 (T5)

### 2.1 依赖体系分析

`mablaslt` 是厂商开发的轻量级矩阵乘加速库，对标业界 CUBLASLt 架构，提供了 GEMM 参数规划、启发式算法选择以及执行句柄管理。其底层构建与依赖体系特征如下：

1. **构建系统架构**:
   - 项目采用现代 CMake (要求版本 3.22+) 进行多层级构建管理。
   - 核心构建逻辑依赖于 `/opt/maps/toolchain/cmake/` 目录下的芯片厂商定制化 CMake 模块：
     - `scale_config.cmake`: 定义芯片型号（如目标 LPU CModel 对应的 target 标识）、系统 include 路径及构建宏。
     - `CMakeDetermineSCALECompiler.cmake` 与 `CMakeDetermineSASMCompiler.cmake`: 注入专用编译器 `scale-cc` 与汇编器 `sasm` 的编译器探测规则。
2. **外部运行时与驱动依赖**:
   - `libmaps.so`: 用户态驱动基础运行时库 (UMD, User Mode Driver)，提供内存分配 (`mapsMalloc`)、流管理 (`mapsStreamCreate`) 与内核启动 (`mapsLaunchKernelEx`)。
   - `libmapskmd.so`: 内核态驱动交互桥梁库，与 `/dev/mfd` 字符设备节点通信。
   - `libmalog.so`: 厂商专用日志追踪与调试输出库。
   - `libllcdsc_kernel.a`: 底层张量描述子与调度加速静态库。
3. **构建模式分支**:
   - 库顶层脚本 `tools/build.sh` 提供了两种构建模式：默认模式面向 SLVS 裸机环境（需要 `libmapsrt.so`）；
   - 面向 QEMU Linux CModel 环境时，必须传递参数 `./tools/build.sh build_umd`，该参数在 CMake 中开启 `-DUSE_MAPS_UMD=ON`，使得编译产物正确链接至 `/opt/maps/lib/libmaps.so`。

### 2.2 硬件拓扑依赖与 4-APC CModel 兼容性

在厂商头文件 `common/device_info.h` 中，整卡硬件常量硬编码定义为：
- `k_die_per_card = 2`
- `k_apc_num_per_die = 32`
- `k_ape_num_per_apc = 4`
- 对应单 Die 具备 32 个 APC (128 APE)，整卡具备 64 个 APC (256 APE)。
- 库内全局初始化例程 `init_sbuf_and_writeadapter` 将硬件网格尺寸设为 `config.gridDim.x = 32`。

针对「该库是否强依赖 32 APC 硬件、在 4-APC CModel 上能否运行」的问题，调研与实测发现：
1. **多核网格划分策略**:
   - `mablaslt` 内部具备基于矩阵尺寸的切分调度器。在处理大矩阵时，网格切分会展开至全部 32 个 APC；
   - 但在小矩阵（$M, N, K \le 256$）或启用 Split-K 切分时，算法根据计算量自动将 `gridDim.x` 约束在较小的核数区间（例如 1 或 4 个 APC）；
2. **CModel 兼容性表现**:
   - 实验证实，只要矩阵瓦片切分后的活跃 APC 索引未越界超出 CModel 实际实例化的 4 个 APC，或者使用小规模测试用例，运行时不会触发非法 APC 访问异常。这使得小规模 GEMM 用例可以在 4-APC 的 CModel 上顺利通过。

### 2.3 实测编译与运行验证

在客体机 QEMU CModel 环境中，加载 `maps` 驱动模块后执行构建与验证：

1. **编译执行**:
   - 执行 `./tools/build.sh build_umd`，顺利编译完成了全部动态库 (`libmablas.so`, `libmablasLt.so`)、单元测试集 (`mablasLt_test`) 以及示例程序 (`00_basic_matmul`, `01_blas_perf`)。
2. **示例程序运行验证 (`00_basic_matmul`)**:
   - 运行针对 $M=256, N=256, K=256$ 的单精度/半精度基础矩阵乘用例；
   - 程序覆盖测试了 4 种转置组合：$A \times B$ (N-N), $A \times B^T$ (N-T), $A^T \times B$ (T-N), $A^T \times B^T$ (T-T)；
   - 客体机终端均成功输出 `Success`，计算精度完全通过参考验证。
3. **性能测试程序运行验证 (`blas_perf`)**:
   - 执行命令 `blas_perf fp16 1 1 256 256 false false`；
   - 运行时成功调用 `mablasLtMatmul` 接口，动态匹配并拉起 JIT 链接的 `gemm_slice_k_matmul_cmode` 内核，顺利完成执行。

### 2.4 源码级硬件周期插桩与实测对比 (T5)

虽然厂商原版 `blas_perf` 依赖宿主机墙钟 API（`mapsEventElapsedTimeEx`，属于已排除计时源），但由于 `mablaslt` 具备完整的 SPU/MPU 内核源码与 CMake 构建树（`./tools/build.sh build_umd`），本项目采用与自主内核 `gemm.sc` 完全一致的硬件周期插桩方法，实现了对厂商官方矩阵乘算子库真实 `device_cycles` 的精确测量。

#### 1. 源码插桩位置与实现方案
所有插桩修改均保存在本地实验环境（`/home/rog/vendor_demo/`），未纳入版本控制：
1. **FP16 内核定位与插桩**:
   - 源文件：`mablaslt/mablasLt/csrc/kernel/general_matmul/general_matmul_mem_bound_slice_k.sc`
   - 入口函数：`gemm_slice_k_matmul_cmode`
   - 插桩方式：引入 `#include "llvm/tick.h"`，在内核执行前调用 `unsigned long long start_tick = tick();`，在所有计算与同步完成退出前调用 `unsigned long long end_tick = tick();`，将差值 `dev_cycles = end_tick - start_tick` 写入输出张量缓冲区首地址（`reinterpret_cast<__DDR unsigned long long*>(h_matD)[0] = dev_cycles`），并通过设备侧 `printf` 打印确认；
2. **FP32 内核定位与插桩**:
   - 源文件：`mablaslt/mablasLt/csrc/kernel/general_matmul/general_matmul_tile.sc`
   - 入口函数：`gemm_matmul_cmode`
   - 插桩方式：采用相同的 `tick()` 捕获机制与缓冲回写方式；
3. **测试前端插桩与重复测量**:
   - 源文件：`mablaslt/test/blas_perf.cpp`
   - 增加热身运行（Warm-up）、测量宿主机墙钟（`std::chrono::high_resolution_clock`）、并直接从 Host 映射区读取设备写回的硬件 `device_cycles`，支持 `--repeats` 重复运行。

#### 2. 实测数据与对比分析 (建档于 `results/vendor_gemm.csv`) [待重新测量]

> [!WARNING]
> **重要方法学预警与重新测量标记**：
> 下表所列厂商 `mablaslt` 周期、吞吐及精度校验字段（`correct` / `max_diff`）已统一标记为 **`[待重新测量 (Pending Re-measurement)]`**。`results/vendor_gemm.csv`（42 行）与诊断数据 `results/vendor_gemm_diagnostic.csv`（2 行）中已内置 `validity=invalid_dma_bypassed` 自描述列，且 `correct` / `max_diff` 列已清空留白。
> 定性诊断实验（固定 $M=8$、将 $K, N$ 从 256 扩展至 512，工作量激增 8 倍，实测执行周期仍为 107,563 / 107,665 恒定不动，落盘于 `results/vendor_gemm_diagnostic.csv`）与源码审计表明：测试程序 `blas_perf` 链接的 `libmablasLt_perf.so` 包含 `-DBLAS_PERF_SHARED_LIB` 宏，底层真实 DMA 数据搬运被彻底宏旁路（No-Op）；且 `blas_perf.cpp` 本身不执行任何 CPU 参考精度校验（此前 CSV 中的 `PASS` / `0.000000e+00` 系采集脚本硬编码填入的非测量占位符）。在 DMA 宏旁路下，真实数据既未搬入 DM 也未写回 Host，完整计算工作并未在真实数据上执行。因此，测得的执行周期仅反映编译期固定分块模板的空转，表内各 $M$ 下吞吐率的“线性增长”完全系代数除法算术表象（$\text{FLOPs} \div \text{常数}$），严禁作为已确认事实引用。

在客体机 QEMU CModel 环境下，针对与自主基准一致的规约尺寸（$K=256, N=256$，$M \in \{1, 2, 4, 8, 16, 32, 64\}$，各 3 次独立测量，`build_id: 36190e5d6e2f`），实测数据汇总如下：

| 数据精度 (dtype) | 批大小 $M$ | 算术强度 $I$ (FLOP/B) | 3 次硬件执行周期 (device_cycles) | 平均吞吐 (FLOPs/cycle) | 相对该精度单 APE 峰值利用率 | 相对 Naive C 算子加速比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP16** (`slice_k`) [待重新测量] | $M=1$ | 1.00 | 51,595 / 51,578 / 51,578 (均值 **51,584**) | **2.541** | 0.124% (基准 2048 FLOPs/cyc) | — (跨精度不直接对比) |
| **FP16** (`slice_k`) [待重新测量] | $M=2$ | 2.00 | 51,538 / 51,455 / 51,486 (均值 **51,493**) | **5.091** | 0.249% (基准 2048 FLOPs/cyc) | — (跨精度不直接对比) |
| **FP16** (`slice_k`) [待重新测量] | $M=4$ | 4.00 | 51,492 / 51,595 / 51,692 (均值 **51,593**) | **10.162** | 0.496% (基准 2048 FLOPs/cyc) | — (跨精度不直接对比) |
| **FP16** (`slice_k`) [待重新测量] | $M=8$ | 8.00 | 52,062 / 52,062 / 51,486 (均值 **51,870**) | **20.216** | 0.987% (基准 2048 FLOPs/cyc) | — (跨精度不直接对比) |
| **FP16** (`slice_k`) [待重新测量] | $M=16$ | 16.00 | 51,540 / 51,595 / 51,692 (均值 **51,609**) | **40.635** | 1.984% (基准 2048 FLOPs/cyc) | — (跨精度不直接对比) |
| **FP16** (`slice_k`) [待重新测量] | $M=32$ | 32.00 | 51,618 / 51,595 / 51,446 (均值 **51,553**) | **81.359** | 3.973% (基准 2048 FLOPs/cyc) | — (跨精度不直接对比) |
| **FP16** (`slice_k`) [待重新测量] | $M=64$ | 64.00 | 51,427 / 51,460 / 51,460 (均值 **51,449**) | **163.047** | 7.961% (基准 2048 FLOPs/cyc) | — (跨精度不直接对比) |
| **FP32** (`tile`) [待重新测量] | $M=1$ | 0.50 | 108,716 / 108,720 / 108,641 (均值 **108,692**) | **1.206** | 0.471% (基准 256 FLOPs/cyc) | **4.33×** (vs Naive C 470,550) [待重测] |
| **FP32** (`tile`) [待重新测量] | $M=2$ | 1.00 | 108,720 / 108,638 / 108,610 (均值 **108,656**) | **2.413** | 0.942% (基准 256 FLOPs/cyc) | **8.44×** (vs Naive C 917,183) [待重测] |
| **FP32** (`tile`) [待重新测量] | $M=4$ | 2.00 | 108,616 / 108,639 / 108,637 (均值 **108,631**) | **4.826** | 1.885% (基准 256 FLOPs/cyc) | **16.67×** (vs Naive C 1,810,449) [待重测] |
| **FP32** (`tile`) [待重新测量] | $M=8$ | 4.00 | 108,717 / 108,643 / 108,636 (均值 **108,665**) | **9.650** | 3.769% (基准 256 FLOPs/cyc) | **33.10×** (vs Naive C 3,596,981) [待重测] |
| **FP32** (`tile`) [待重新测量] | $M=16$ | 8.00 | 108,717 / 108,642 / 108,614 (均值 **108,658**) | **19.301** | 7.539% (基准 256 FLOPs/cyc) | — (无 Naive C M=16 基线) |
| **FP32** (`tile`) [待重新测量] | $M=32$ | 16.00 | 108,716 / 108,639 / 108,610 (均值 **108,655**) | **38.602** | 15.079% (基准 256 FLOPs/cyc) | — (无 Naive C M=32 基线) |
| **FP32** (`tile`) [待重新测量] | $M=64$ | 32.00 | 108,615 / 108,616 / 108,720 (均值 **108,650**) | **77.207** | 30.159% (基准 256 FLOPs/cyc) | — (无 Naive C M=64 基线) |

#### 3. 核心机制剖析与 DMA 宏旁路调查

1. **定性诊断实验与恒定周期根因排查（数据落盘于 `results/vendor_gemm_diagnostic.csv`）**:
   - **恒定周期与抖动表象**：初测在 $M \in [1, 64]$ 内测得 FP32 执行周期恒为 ~108.6k 周期，FP16 恒为 ~51.5k 周期，且同配置 3 次重复存在 $\pm 80\sim 110$ 周期抖动（与确定性模拟器中自主内核的零方差特征不一致）；
   - **定性实验结果**：为检验周期是否受问题规模控制，执行了定性诊断实验：保持 $M=8$，将 $K$ 与 $N$ 从 256 翻倍至 512（浮点运算量增加 8 倍，访存量增加 4 倍）。在 QEMU CModel 上的实测结果（已归档于 `results/vendor_gemm_diagnostic.csv`）表明：
     - $K=N=256$: `device_cycles` 均值为 108,665 周期；
     - $K=N=512$: `device_cycles` 实测为 **107,563（Warm-up）/ 107,665（Repeat 0）周期**；
     - **判定**：运算量扩大 8 倍，执行周期完全未发生 8 倍增长，反而恒定在同一常数水平。这证明测得周期与 $M, K, N$ 全部解耦，排除了所谓“大分块瓦片已满载”的解释；
   - **源码根因与推理更正（`correct` / `max_diff` 同样无效）**：
     - `blas_perf` 链接的 `libmablasLt_perf.so` 开启了 `-DBLAS_PERF_SHARED_LIB` 宏，在 `llcd_csu_dma.h:48` 与 `llcd_dma.h:306` 中通过条件编译将所有 `deviceMemcpy*` 宏旁路（No-Op），未执行真实 DMA；
     - 调度的微内核接收固定编译期常量，其 $\pm 100$ 周期抖动来自 `SYNC_MPU_KERNEL` barrier 轮询；
     - **推理更正与精度字段还原**：此前曾以 `max_diff=0`、`correct=PASS` 作为“计算工作已执行但落在计时区间外”的论据，**该论据不成立**——源码核查表明 `blas_perf.cpp` 从未执行 CPU 参考校验（仅 `00_basic_matmul` 含精度校验），此前 CSV 中的 `PASS` 与 `0.000000e+00` 系采集脚本硬编码填入的占位符；在 DMA 宏旁路下，真实数据的搬入、计算与写回根本没有执行。结论不变，但周期恒定的理由统一归结为 **DMA 宏旁路与编译期固定瓦片空转**；
   - **指标禁用、CSV 留白与内置失效标记**：
     此前基于该固定周期所作的推论——包括：“$M_{tile} \ge 64$”`[待重新测量]`、“$M=1$ 浪费 98.44%”`[待重新测量]`、“饱和吞吐 77.21 FLOPs/cycle”`[待重新测量]`、“264.9× 相对自研 Naive C 提速”`[待重新测量]`、“3.32× 官方未释放开销”`[待重新测量]`、“33.10× 相对加速比”`[待重新测量]`、以及 `correct` / `max_diff` 精度校验结果`[待重新测量]`，**均已标记为待重新测量，严禁作为已确立结论引用**。两份 CSV 内部已清空 `correct` / `max_diff` 并新增 `validity=invalid_dma_bypassed` 列。
2. **严格区分精度口径与 FP16 真实算力瓶颈 [待重新测量]**:
   - FP16 与 FP32 硬件运算单元与理论峰值存在本质差异（FP16 峰值为 2048 FLOPs/cycle，是 FP32 256 FLOPs/cycle 的 8 倍），跨精度不可直接计算加速比；
   - 实测中厂商 FP16 内核执行周期约 51.5k 周期，因同样属于 mock DMA 区域，其 163.05 FLOPs/cycle 吞吐与 7.96% 峰值利用率均属 `[待重新测量]`，后续须在真实带 DMA 运行时中建立准确计时。
3. **878× 理论物理差距的两层分解与正交维度辨析**:
   - 在 FP32 同精度口径下，理论峰值（256 FLOPs/cycle）与自研 Naive C（0.2915 FLOPs/cycle @ M=8）的差距为 **878×**（严格基于硬件三层物理分解：$8.00\times \text{发射} \times 74.00\times \text{编译器缺乏流水} \times 1.481\times \text{控制} = 878\times$）；
   - **代数乘法链修正**：严禁将瓦片空载因子（如 64×）与提速因子和未释放因子做三项连乘（$264.9\times \times 64\times \times 3.32\times = 56,300\times \neq 878\times$）。在理论分解上，两层乘积关系为：
     $$\text{总物理差距 (878×)} = \text{厂商优化相对 Naive C 提速 (264.9× [待重测])} \times \text{厂商优化相对物理峰值差距 (3.316× [待重测])}$$
     其乘积严格满足 $264.9 \times 3.316 \approx 878$。
   - **瓦片空载的正交解释地位**：若后续在真实调度中确认存在大瓦片，瓦片未充分填充（如小 Batch 场景）属于解释为何单 Batch 未达饱和吞吐的**独立正交维度**，不混入 878× 主干物理差距乘积中。

---

### 2.5 真实库 `00_basic_matmul` 入口核验、全程插桩与 T3 单调性门禁测试 (T1–T3)

为彻底排除 `libmablasLt_perf.so` 中 `-DBLAS_PERF_SHARED_LIB` 宏对 DMA 的旁路影响，本轮按照任务书规范切换至 `sample/mablasLt/00_basic_matmul` 入口，依次执行 T1（入口与链接目标确认）、T2（覆盖完整 kernel 的计时插桩）与 T3（单调性门禁测试）。

#### 1. T1：确认入口与链接目标

1. **`00_basic_matmul` 实际链接目标核验 (`libmablasLt.so`)**：
   - 核查 `mablaslt/sample/mablasLt/CMakeLists.txt` 第 8–11 行：
     ```cmake
     target_link_libraries(${obj_path} ${MAPS_LIB} -lmablasLt)
     target_include_directories(${obj_path} PUBLIC ${MABLASLT_INC_PATHS})
     target_link_directories(${obj_path} PUBLIC ${MABLASLT_LIB_PATHS})
     add_dependencies(${obj_path} mablasLt)
     ```
     确认构建目标显式依赖并链接真实动态库 `mablasLt`（`libmablasLt.so`），而非 `mablasLt_perf`。
   - 在 QEMU CModel 客体机对编译产物执行 `ldd /home/emu/vendor_demo/demo4scalee2e/mablaslt/build/sample/mablasLt/00_basic_matmul`，输出结果为：
     ```text
     libmaps.so => /opt/maps/lib/libmaps.so
     libmablasLt.so => /home/emu/vendor_demo/demo4scalee2e/mablaslt/build/mablasLt/libmablasLt.so
     libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6
     ```
     确认运行期加载的是真实库 `libmablasLt.so`（MD5 `build_id = 41a1645b9e143f255310c8b65a0c798b`）。

2. **编译参数是否含 `-DBLAS_PERF_SHARED_LIB` 核验**：
   - 核查 `mablaslt/mablasLt/CMakeLists.txt` 第 36 行与第 71 行：
     - `build_spu_obj(spu_objs "${SPU_SRCS}" "${MABLASLT_INC_PATHS}" "mpu_objs")` 编译设备侧 `.sc` 源文件时，仅注入 `-DBLAS_SHARED_LIB -DMAPS_DEVICE_COMPILER`；
     - `build_libs(mablasLt mablaslt_objs comm_objs spu_objs)` 打包生成 `libmablasLt.so`，全程**不含** `-DBLAS_PERF_SHARED_LIB`（该宏仅在第 55 行 `build_perf_spu_obj` 与第 78 行 `mablasLt_perf` 中定义）。
   - 对 `libmablasLt.so` 解包设备侧符号确认：`llcd_csu_dma.h:48` 与 `llcd_dma.h:306` 中的真实 `deviceMemcpyPaddingAsync` / `deviceMemcpyAsync` 在 `libmablasLt.so` 中均完整编译并生效。

3. **精度校验机制与真实 `max_diff` 输出确认**：
   - 核查 `sample/mablasLt/00_basic_matmul.cpp` 与 `sample/common/utils.h`：
     - 原程序通过 `host_compute`（基于 `half_float::half` 在 CPU 侧执行三层循环 $C_{ref} = \alpha A B + \beta C$）计算参考真值，并调用 `check_result(h_C, h_C_ref, m, n, k)` 逐元素比对误差（阈值 `tolerance = k * 1e-5f`）；
     - 为支持 FP32 与 FP16 双精度下的真实 `max_diff` 采集，在 `utils.h` 中增加了 `host_compute_fp32`（FP32 CPU 双精度累加参考实现），并扩展 `check_result` 使其遍历全部 $M \times N$ 个输出元素，精确统计最大绝对误差 `max_diff = max_i |h_C[i] - h_C_ref[i]|`，在终端打印 `correct=PASS/FAIL` 与 `max_diff=%e`。

#### 2. T2：覆盖完整 Kernel 的 `tick()` 插桩位置

在本地实验目录 `/home/rog/vendor_demo/demo4scalee2e/mablaslt/`（未纳入 git 仓库）中，对真实库调用的内核及 Host 侧缓冲执行如下插桩：

1. **FP32 通用矩阵乘内核 (`mablasLt/csrc/kernel/general_matmul/general_matmul_tile.sc`)**：
   - **函数名**：`gemm_matmul_cmode`（第 70–120 行）
   - **起点插桩位置**：在函数入口处完成基础栈变量声明后，先执行 `__syncthreads()` 对齐单 APC 内 4 个 APE 线程，紧接着在 **`matmulkernel.init(...)`（第 91 行）与 `matmulkernel.process()`（第 94 行）之前**（第 84 行）读取 `unsigned long long t_start = tick();`；
   - **终点插桩位置**：在 `matmulkernel.process()` 完成全部外层 `b_loops * m_loops * n_loops * k_loops` 瓦片迭代（含首次 `LlcdCsuDmaReadMatA/B` 预取、循环内双缓冲 DMA 搬运、MPU 微码核 `launch_sync` 计算、以及最后一次 `write_out_to_global` 写回 DDR 并在内部完成 `LlcdDmaWait`）之后，执行 `__syncthreads()` 并由 `threadIdx.x == 0 && blockIdx.x == 0` 读取 `unsigned long long t_end = tick();`（第 113 行）；
   - **无损回写通道**：将 `dev_cycles = t_end - t_start` 写入输出矩阵末尾越界预留区 `reinterpret_cast<__DDR unsigned long long*>(h_matD + tilingGM.B * tilingGM.M * tilingGM.N)[0] = dev_cycles;`（第 115 行）。Host 侧在 `00_basic_matmul.cpp` 中为矩阵 `C` 和 `h_C` 额外多分配 8 个元素（`(m * n + 8) * sizeof(float)`），从而在不破坏前 $M \times N$ 个输出矩阵元素、完全不影响 `check_result` 精度校验的前提下读取 `dev_cycles`。

2. **FP16 访存受限 Slice-K 内核 (`mablasLt/csrc/kernel/general_matmul/general_matmul_mem_bound_slice_k.sc`)**：
   - **函数名**：`gemm_slice_k_matmul_cmode`（第 68–103 行）
   - **插桩位置**：同样在 `__syncthreads()` 后、`matmulkernel.init(...)` 与 `matmulkernel.process()` 之前记录 `t_start = tick()`，在 `matmulkernel.process()` 与 `__syncthreads()` 之后记录 `t_end = tick()`，并将差值写回 `reinterpret_cast<__DDR unsigned long long*>(h_matD + tilingGM.B * tilingGM.M * tilingGM.N)[0]`。

#### 3. T3：单调性门禁（Monotonicity Gate）实测结果与源码根因剖析

根据任务书 T3 规范，在批量采集前使用真实库 `00_basic_matmul`（`libmablasLt.so`，`build_id = 41a1645b9e143f255310c8b65a0c798b`）运行三组单调性门禁测试与一组 Split-K 诊断测试，全部原始数据已落盘至 `results/vendor_gemm_real_gate.csv`：

| 测试组别 | 矩阵规模 $(M, K, N)$ | 精度 | 3 次重复 `device_cycles` | 均值 (极差) | 相对基线比值 | `correct` | `max_diff` | `validity` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **基线组** | $(8, 256, 256)$ | FP32 | `76,076 / 75,969 / 75,996` | **76,013.67** (`107`) | `1.0000×` | `PASS` | `0.000000e+00` | `valid_with_explanation` |
| **$K,N$ 翻倍门禁组** | $(8, 512, 512)$ | FP32 | `77,909 / 77,991 / 77,911` | **77,937.00** (`82`) | **`1.0253×`** (+1,923.33 cyc) | `PASS` | `0.000000e+00` | `valid_with_explanation` |
| **$M$ 翻倍交叉验证组** | $(16, 256, 256)$ | FP32 | `76,241 / 76,239 / 76,240` | **76,240.00** (`2`) | **`1.0030×`** (+226.33 cyc) | `PASS` | `0.000000e+00` | `valid_with_explanation` |
| **$M=512$ 验证组** | $(512, 256, 256)$ | FP32 | `81,705 / 81,744 / 81,709` | **81,719.33** (`39`) | **`1.0751×`** (+5,705.67 cyc) | `PASS` | `0.000000e+00` | `valid_with_explanation` |
| **$N=1024$ 验证组** | $(8, 256, 1024)$ | FP32 | `75,996 / 76,076 / 75,970` | **76,014.00** (`106`) | **`1.0000×`** (+0.33 cyc) | `PASS` | `0.000000e+00` | `valid_with_explanation` |
| **Split-K 边界组** | $(8, 1024, 512)$ | FP32 | `77,911` (1 repeat) | **77,911.00** | `1.0250×` | `FAIL` | `1.007412e+03` | `invalid_split_k_unreduced_workspace` |

##### 门禁判定结论：**通过（`valid_with_explanation`）**

切换至真实库 `libmablasLt.so` 后的有效性与源码级机制核验如下：
1. **真实 DMA 与单调正相关增长**：
   - 注：`libmablasLt.so` 与 `libmablasLt_perf.so` 两个构建的编译宏与插桩边界均不同，两组周期不具可比性，不作对比；
   - 当 $M$ 从 $8 \to 16 \to 512$ 时，单 APC 周期严格单调递增（`76,013.67 -> 76,240.00 -> 81,719.33`），反映单 APC 真实 DDR $\leftrightarrow$ DM DMA 载荷从 8 行增至 16 行再增至满瓦片 256 行（$+226.33$ 周期与 $+5,705.67$ 周期）；当 $K, N$ 从 $256 \to 512$ 时，周期严格增加 **+1,923.33 周期**（`76,013.67 -> 77,937.00`）；
2. **物理峰值合理性旁证（86.2% 单 APC 峰值）**：
   - 在 $(M=8, K=256, N=256)$ 下，内核启动 1 个 APC（4 个 APE），单个 $256 \times 256 \times 512$ 瓦片的实际硬件浮点运算量为 $2 \times 256 \times 256 \times 512 = 67.11\text{ MFLOP}$，除以实测周期 `76,014` 得单 APC 硬件实际吞吐 **`882.9 FLOPs/cycle`**，达到单 APC FP32 理论峰值（$4 \times 256 = 1024\text{ FLOPs/cycle}$）的 **86.2%**（若按 MPU 纯计算段 ~74,000 周期计则达 **88.6%**）；
   - 在 $(M=512, K=256, N=256)$ 下，调度器启动 **2 个并行 APC**（`m_blk_num = 2`，共 8 个 APE，双 APC 理论峰值 $2 \times 1024 = 2048\text{ FLOPs/cycle}$），实测周期 `81,719.33`：
     - **有用功口径（$K=256$）**：双 APC 合计有用功为 $2 \times 512 \times 256 \times 256 = 67.11\text{ MFLOP}$，双 APC 合计有用功吞吐为 $67.11\text{ MFLOP} \div 81,719.33 = \mathbf{821.2\text{ FLOPs/cycle}}$（单 APC 均摊有用功吞吐为 `410.6 FLOPs/cycle`），相对双 APC 峰值 `2048 FLOPs/cycle`（或单 APC 有用功 `410.6` 相对单 APC 峰值 `1024`）的利用率为 **$821.2 \div 2048 = 410.6 \div 1024 = \mathbf{40.10\%}$**；
     - **硬件实际功口径（含 $K_{tile}=512$ 零填充空转）**：2 个 APC 合计执行的硬件浮点运算量为 $2 \times (2 \times 256 \times 256 \times 512) = 134.22\text{ MFLOP}$，双 APC 合计硬件实际吞吐为 $134.22\text{ MFLOP} \div 81,719.33 = \mathbf{1642.4\text{ FLOPs/cycle}}$（单 APC 硬件实际吞吐为 `821.2 FLOPs/cycle`），相对双 APC 峰值 `2048 FLOPs/cycle`（或单 APC 硬件实际吞吐 `821.2` 相对单 APC 峰值 `1024`）的利用率为 **$1642.4 \div 2048 = 821.2 \div 1024 = \mathbf{80.20\%}$**；
3. **真实 CPU 精度校验生效**：
   - $K \le 512$ 的全部 15 行测试在真实 DMA 搬入与写回下均通过 CPU 双精度参考比对（`correct=PASS`, `max_diff=0.000000e+00`），而 $(8, 1024, 512)$ 触发跨 APC Split-K 时真实检出未归约 workspace 误差（`correct=FAIL`, `max_diff=1.007412e+03`，属 `00_basic_matmul` 入口未调用二级 Split-K 规约 kernel 的使用边界，非库缺陷）。

经对 `mablasLt` 源码逐层追踪，$(8, 512, 512)$、$(512, 256, 256)$ 与 $(8, 256, 1024)$ 单 APC `tick()` 周期未成倍翻番的底层机制如下（均有明确源码行号支撑）：

1. **启发式算法选择桩函数（Stub）无条件选中 `B1M256N256K512` 大瓦片 (`general_matmul_register.cpp:215-226`)**：
   - `mablasLt` 为 FP32 注册了两个通用矩阵乘算子：`general_matmul_fp32_B1M256N256K256_op`（$K_{tile}=256$）与 `general_matmul_fp32_B1M256N256K512_op`（$K_{tile}=512$）；
   - 但其性能预测函数 `MABLASLT_IMPL_KERNEL_PREDICT_PERF` 尚未实现（源码注释：`// TODO(Jiahao): add perfInfo`），将前者的预测耗时硬编码为 `perfInfo.time_ms = 1e-8`，后者硬编码为 `perfInfo.time_ms = 1e-9`；
   - 由于 `1e-9 < 1e-8`，启发式选择器在任何 $(M, K, N)$ 规模下**无条件选中 `B1M256N256K512`**（单瓦片固定为 $M_{tile}=256, N_{tile}=256, K_{tile}=512$）。

2. **微码核编译期硬编码 $K_{tile}=512$ 导致 $K=256$ 与 $K=512$ 的 MPU 计算周期完全相同 (`general_matmul_kernel.h:75,83,346-352`)**：
   - 在 `MatmulTile<1, 256, 256, 512, 1, 2, 2, 1>` 中，单 APC 内的 4 个 APE 按 $2 \times 2$ 网格切分 $256 \times 256$ 输出瓦片，每个 APE 固定负责 $M_{thd}=64, N_{thd}=64, K_{thd}=512$；
   - 在 `MatmulKernel::compute()` 中，调用底层汇编微码核 `matmul_mpu_kernel_.launch_sync(..., dev_tile.M_THD_TILE_A, dev_tile.N_THD_TILE_B, dev_tile.K_THD_TILE_A, ...)` 传入的维度参数 `64, 64, 512` 全部是**编译期 `constexpr` 常量**；
   - 当运行期 $K=256 < 512$ 时，`LlcdCsuDmaReadMatA/B` 将 DM 中 $K \in [256, 512)$ 区域补零（`DmaPaddingType::ZERO`），而汇编微码核 `general_matmul_fp32.m.asm` 依然完整执行 512 步 K 循环（耗时约 74,000 周期）。因此，$K$ 从 $256$ 增至 $512$ 时，`k_loops_ = CEIL_DIV(K, 512)` 均为 `1`，**MPU 计算周期增量为 0**，仅增加真实 DMA 传输周期。

3. **多 APC 网格切分自动将 $M$ 与 $N$ 方向瓦片分摊至多个并行 APC (`general_matmul_info.h:113-141`, `device_info.h:16`)**：
   - 在 `GeneralMatmulInfo<1, 256, 256, 512>::get_mm_tile_split_k` 中，当总瓦片数 $\lceil B/1 \rceil \times \lceil M/256 \rceil \times \lceil N/256 \rceil \times \lceil K/512 \rceil \le 32$（`DeviceInfo::k_apc_num_per_die = 32`）时，调度器直接令 `mm_tile.m_blk_num = max_m_apc_num = CEIL_DIV(M, 256)` 且 `mm_tile.n_blk_num = max_n_apc_num = CEIL_DIV(N, 256)`；
   - 因此在 $(8, 512, 512)$ 下 `n_blk_num = 2`（2 个 APC 并行），在 $(512, 256, 256)$ 下 `m_blk_num = 2`（2 个 APC 并行），在 $(8, 256, 1024)$ 下 `n_blk_num = 4`（4 个 APC 并行），每个 APC 内部依然只执行 `m_loops_ = 1, n_loops_ = 1, k_loops_ = 1`（`pipeline_loops_ = 1`）轮瓦片计算，单 APC 的 `device_cycles` 增量严格对应单 APC 内真实 DMA 传输载荷的增长。
   - 此外，当测试 $(8, 1024, 512)$ 时，`k_blk_num = 2` 触发跨 APC Split-K 并行切分（`gridDim.x = 4`），而 `00_basic_matmul.cpp` 仅调用了主矩阵乘算子、未调用后续 Split-K 规约算子，导致输出停留在 workspace 中（`correct=FAIL, max_diff=1.007412e+03`）。

#### 4. T4：真实库（方案 A，原版未修改 `libmablasLt.so`）批量采集实测结果

基于 T3 门禁通过（`valid_with_explanation`）及方案 A（保持原版未修改 `libmablasLt.so` as-shipped 行为），在同二进制（`build_id = 41a1645b9e143f255310c8b65a0c798b`, `commit_hash = 057d850`）下完成了 $K=256, N=256, M \in \{1, 2, 4, 8, 16, 32, 64\}$ 的 FP32 全序列实测（各 3 次重复，共 21 行，完整归档于 `results/vendor_gemm_real.csv`，`validity=valid`）：

| 批大小 $M$ | 3 次重复 `device_cycles` | 均值 (极差) | 相对 $M=4$ 增量 | 4-APE 聚合吞吐 (FLOPs/cycle) | 单 APE 归一化吞吐 (FLOPs/cyc/APE) | 相对单 APC 峰值 (1024) | 相对自研 Naive C 加速比 (单 APE 归一化 / 端到端含 4× 核数) | `correct` / `max_diff` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M=1$** | `75,954 / 75,804 / 75,954` | **75,904.00** (`150`) | *抖动区间*\* | 1.727 | 0.432 | **0.169%** | **1.55×** / **6.20×** (vs 470,550) | `PASS` / `0.00e+00` |
| **$M=2$** | `75,804 / 75,804 / 75,804` | **75,804.00** (`0`) | *抖动区间*\* | 3.458 | 0.865 | **0.338%** | **3.03×** / **12.10×** (vs 917,183) | `PASS` / `0.00e+00` |
| **$M=4$** | `75,805 / 75,832 / 75,780` | **75,805.67** (`52`) | Baseline ($M=4$) | 6.916 | 1.729 | **0.675%** | **5.97×** / **23.88×** (vs 1,810,449) | `PASS` / `0.00e+00` |
| **$M=8$** | `76,076 / 75,969 / 75,996` | **76,013.67** (`107`) | `+208.00` | 13.795 | 3.449 | **1.347%** | **11.83×** / **47.32×** (vs 3,596,981) | `PASS` / `0.00e+00` |
| **$M=16$** | `76,241 / 76,239 / 76,240` | **76,240.00** (`2`) | `+434.33` | 27.507 | 6.877 | **2.686%** | — (无 Naive C M=16 实测) | `PASS` / `0.00e+00` |
| **$M=32$** | `76,825 / 76,827 / 76,825` | **76,825.67** (`2`) | `+1,020.00` | 54.595 | 13.649 | **5.332%** | — (无 Naive C M=32 实测) | `PASS` / `0.00e+00` |
| **$M=64$** | `78,179 / 78,096 / 78,099` | **78,124.67** (`83`) | `+2,319.00` | 107.375 | 26.844 | **10.486%** | *92.08× / 368.33× (基于自研 M=8 吞吐外推，未实测)* | `PASS` / `0.00e+00` |
| *满瓦片参照 ($M=512$, `m_blk_num=2` 双 APC)* | `81,705 / 81,744 / 81,709` | **81,719.33** (`39`) | `+5,913.67` | **有用功 ($K=256$)**: 双 APC 821.2 (单 APC 410.6)<br>**硬件功 ($K=512$)**: 双 APC 1642.4 (单 APC 821.2) | 有用功: 102.65<br>硬件功: 205.30 | **有用功**: `821.2÷2048` = **40.10%**<br>**硬件功**: `1642.4÷2048` = **80.20%** (对双 APC 峰值 2048) | — | `PASS` / `0.00e+00` |

*\*表注：$M=1, 2, 4$ 三点的均值差异（`75,904.00 / 75,804.00 / 75,805.67`，最大差 100 周期）完全落在同配置重复抖动范围内（如 $M=1$ 组内 3 次重复极差为 150 周期，$M=1$ 均值甚至略高于 $M=2$），这三点不构成单调趋势、不作解读；严格单调递增自 $M=4$ 起成立。此外，厂商 kernel 固定占用 1 个 APC（4 个 APE，每个 APE 线程瓦片 $M_{thd}=64, N_{thd}=64$，共同覆盖 $256 \times 256$ 输出瓦片），故表中 `4-APE 聚合吞吐` 须对标单 APC 峰值 `1024 FLOPs/cycle`（等价于 `单 APE 归一化吞吐` 对标单 APE 峰值 `256 FLOPs/cycle`）。*

**核心规律总结**：
1. **自 $M=4$ 起真实 DMA 搬运增量清晰单调**：在 $M=1, 2, 4$ 极小载荷下，1~4 行矩阵的 DMA 传输差异被 barrier 轮询抖动（$\le 150$ 周期）淹没，不构成单调趋势；自 $M=4$ 起，随着 $M$ 每次翻倍（$4 \to 8 \to 16 \to 32 \to 64 \to 256$），`device_cycles` 呈严格单调递增（`75,805.67 -> 76,013.67 -> 76,240.00 -> 76,825.67 -> 78,124.67 -> 81,719.33`），相邻档位增量分别为 `+208.00`、`+226.33`、`+585.67`、`+1,299.00` 与 `+3,594.67` 周期，准确映射矩阵 $A$ 与 $D$ 的真实 DDR $\leftrightarrow$ DM DMA 载荷增长；
2. **单 APE 归一化加速比（11.83×）与小 Batch 瓦片容量浪费（74.2× 未释放）**：
   - 由于原版 `libmablasLt.so` 在 $M \le 256$ 下固定启动 1 个 APC（4 个 APE）协同执行 $M_{tile}=256, N_{tile}=256, K_{tile}=512$ 大瓦片，而自研 `gemm.sc` 仅使用 1 个 APE，在 $M=8, K=256, N=256$ 下厂商 4-APE 聚合吞吐 `13.795 FLOPs/cycle` 折合 **单 APE 归一化吞吐为 `13.795 ÷ 4 = 3.4487 FLOPs/cycle`**（占单 APC 峰值 1024 或单 APE 峰值 256 的 **1.347%**）；
   - 对照自研单核 Naive C（`0.2915 FLOPs/cycle`），**手写汇编微码与指令调度在单 APE 口径下真实带来的提速为 `3.4487 ÷ 0.2915 = 11.83×`**（而 `47.32×` 为包含 4 倍硬件规模差异的端到端周期加速比）；
   - 厂商单 APE 吞吐（`3.4487`）距单 APE 理论峰值（`256`）仍差 **`256 ÷ 3.4487 = 74.23×`**（$11.83 \times 74.23 \approx 878\times$），该未释放部分绝大部分源于固定瓦片与小规模输入的严重不匹配：$M=8$ 对 $M_{tile}=256$ 浪费 **96.875%** 的 $M$ 维容量（损失 $32\times$），且缺陷 2 使 $K=256$ 对 $K_{tile}=512$ 浪费 **50%** 的 $K$ 维容量（损失 $2\times$），两项合计达 $64\times$，其余 $74.23 \div 64 = 1.160\times$ 为满瓦片微码核本身的 DMA 与流水余量（对应满瓦片 86.2% 峰值利用率）。

---

### 2.6 厂商工具链缺陷记录

在 T1–T3 源码审查与 CModel 实测过程中，定位并量化了厂商算子库与测试工具链中的两项独立软件缺陷（均保持原样记录，不修改厂商库算法逻辑）：

#### 缺陷 1：`blas_perf` (`libmablasLt_perf.so`) 编译宏旁路全部真实 DDR $\leftrightarrow$ DM DMA 搬运

- **现象**：
  使用厂商性能测试入口 `blas_perf` 测试 GEMM 时，无论矩阵规模 $(M, K, N)$ 如何变化（如 $M \in \{1..64\}, K=N=256$ 乃至 $K=N=512$ 计算量扩大 8 倍），FP32 的 `device_cycles` 恒定在 `~108,650` 周期（`107,563 ~ 108,665`），FP16 恒定在 `~51,500` 周期，周期与问题规模 $(M, K, N)$ 完全解耦，且 `blas_perf` 不执行任何输出正确性比对。
- **源码位置**：
  - `sample/mablasLt/CMakeLists.txt:21`：`blas_perf` 显式链接 `mablasLt_perf`（`target_link_libraries(blas_perf mablasLt_perf ...)`）；
  - `mablasLt/CMakeLists.txt:153`：`mablasLt_perf` 目标注入了 `-DBLAS_PERF_SHARED_LIB` 编译宏（`target_compile_definitions(mablasLt_perf PUBLIC BLAS_PERF_SHARED_LIB)`）；
  - `common/include/llcd_csu_dma.h:48` 与 `common/include/llcd_dma.h:306`：底层 `LlcdCsuDmaReadMatA`、`LlcdCsuDmaReadMatB` 与 `LlcdDma1dWrite` 中的真实异步 DMA 调用（`deviceMemcpyPaddingAsync` / `deviceMemcpyAsync`）全部被包裹在 `#ifndef BLAS_PERF_SHARED_LIB` 条件编译块内。
- **可量化后果**：
  在 `libmablasLt_perf.so` 下，内核从 DDR 到片上 DM 的数据搬入及从 DM 到 DDR 的写回全部退化为空操作（No-Op），内核仅在未初始化的片上内存上空转固定的编译期微码循环。基于该入口测得的一切周期数据均不反映真实硬件访存与边界处理开销（已全量归档并标记 `validity=invalid_dma_bypassed`）。
- **复现方式**：
  在 CModel 中运行 `blas_perf` 对比 $(M=8, K=256, N=256)$ 与 $(M=8, K=512, N=512)$（原始复现数据见 `results/vendor_gemm_diagnostic.csv`），工作量增加 8 倍而 `device_cycles` 分别为 `108,665` 与 `107,631`。

#### 缺陷 2：`mablasLt` 启发式性能预测函数未实现（`TODO` 桩函数），导致 $K \le 256$ 任务被无条件派发至 $K_{tile}=512$ 内核

- **现象**：
  使用真实库 `libmablasLt.so`（`00_basic_matmul`）测试 $K=256$ 的 FP32 矩阵乘（如 $M \in \{1..64\}, K=256, N=256$）时，尽管库内已编译并注册了专门针对 $K=256$ 的 `B1M256N256K256` 内核（`MatmulTile<1, 256, 256, 256, ...>`），运行期日志显示 `mablasLtMatmulAlgoGetHeuristic` 始终派发 $K_{tile}=512$ 的 `B1M256N256K512` 内核（`MatmulTile<1, 256, 256, 512, ...>`）。
- **源码位置**：
  - `mablasLt/csrc/op_register/general_matmul_register.cpp:215-226`：
    在 `general_matmul_fp32_B1M256N256K256_op` 与 `general_matmul_fp32_B1M256N256K512_op` 的注册代码中，启发式性能预测宏 `MABLASLT_IMPL_KERNEL_PREDICT_PERF` 均留有 `// TODO(Jiahao): add perfInfo` 注释，未根据输入矩阵维度 $(M, K, N)$ 计算预测耗时，而是将 `B1M256N256K256` 硬编码为 `perfInfo.time_ms = 1e-8`，将 `B1M256N256K512` 硬编码为 `perfInfo.time_ms = 1e-9`；
  - `mablasLt/csrc/kernel/general_matmul/general_matmul_kernel.h:75,83,346-352`：
    `MatmulKernel::compute()` 向底层汇编微码核 `matmul_mpu_kernel_.launch_sync` 传入编译期常量 `K_THD_TILE_A = 512`。
- **可量化后果**：
  由于 `1e-9 < 1e-8` 恒成立，所有 $K=256$ 的矩阵乘请求均被无条件路由至 $K_{tile}=512$ 内核：DMA 阶段对 $K \in [256, 512)$ 补零，MPU 汇编微码核强制执行完整的 512 步 K 循环（约 74,000 周期），导致 **$K=256$ 任务在 K 维方向被强制垫入 50% 的零填充无效计算，K 维有效算力利用率上限被直接折半（最多 50%）**。
- **复现方式**：
  在 CModel 中运行链接 `libmablasLt.so` 的 `00_basic_matmul` 测试 $(M=8, K=256, N=256)$ 与 $(M=8, K=512, N=512)$（见 `results/vendor_gemm_real_gate.csv`）：运行时日志打印启动内核符号均为 `MatmulTileILi1ELi256ELi256ELi512...`，且 $K=256$（`76,013.67` 周期）与 $K=512$（`77,937.00` 周期）的 MPU 计算段周期完全一致。

---

## 3. maops 算子库与 FlashAttention 实现调研 (T6)

### 3.1 算子库实现形态分布

调研查阅了 `maops` 算子库的官方功能支持矩阵清单文件 `doc/op_list.md`。该文件统计了全部 30 余种大模型高频算子的实现形态。

**核心发现**:
- 清单中对每一个受支持的算子均列出了 `Assembly Op`（手工汇编实现）与 `Intrinsic Op`（C/C++ Intrinsic 高级语言实现）两列；
- **统计结果呈现压倒性的单向倾斜**:
  - `flash_attention`: Assembly Op 列标记 `√`，Intrinsic Op 列为空；
  - `matmul`: Assembly Op 列标记 `√`，Intrinsic Op 列为空；
  - `layernorm` / `rmsnorm`: Assembly Op 列标记 `√`，Intrinsic Op 列为空；
  - `silu` / `gelu`: Assembly Op 列标记 `√`，Intrinsic Op 列为空；
  - `softmax`: Assembly Op 列标记 `√`，Intrinsic Op 列为空；
  - `rope` (旋转位置编码): Assembly Op 列标记 `√`，Intrinsic Op 列为空；
- **全库所有标注为支持状态的算子，100% 依赖 Assembly Op，Intrinsic Op 的可用数量为 0**。

### 3.2 FlashAttention 内核实现剖析

进一步深入检查 `maops/third_party/flashattention/` 及其内核源码目录：
1. **SPU 标量控制层 (`kernels/spu/flash_attention_spu/`)**:
   - 该目录下包含 `flash_attention_spu.sc` 等 C++ 源文件；
   - 经审查，SPU 源文件完全不包含任何 MPU 向量浮点计算逻辑，其主要职责仅为解析来自 Host 的 Tensor 描述子、计算分块偏移、配置片上 DM 地址、以及向任务队列提交 MPU 计算任务。
2. **MPU 向量计算核心层 (`kernels/mpu/flash_attention_mpu/`)**:
   - 该目录下的全部算子计算实现均为纯手工编写的 VLIW 微码汇编源文件（`*.m.asm`）；
   - 其中最为核心的融合注意力计算内核 `mpu_fa_fused_exp_rowsum_pvgemm_diagmul_v02.m.asm`，文件体积达到 **495 KB**，包含数千行经过精细静态排布的 VLIW 指令束（Instruction Packets）；
   - 汇编代码中显式进行了微码发射槽位规划、向量寄存器重命名、以及乘累加累加器 (`FMAMR`) 的全流水展开。

### 3.3 架构发现与软件生态启示

上述调查结果具有极为深刻的体系结构与软件工程启示：

1. **印证厂商性能反馈与编译器瓶颈**:
   - 官方算子库 100% 采用手工汇编的事实，完全印证了此前厂商书面记录中的核心陈述：“**c 的性能比微码差很多**”；
   - 结合本项目在单向量 FMA 算子上的测试（`scale-cc` 编译 C 语言耗时 74 周期/op，而手写汇编仅需 1.003 周期/op，存在约 74 倍开销放大），说明厂商工程团队自身在开发核心算子库时，同样无法依赖现有的 C 编译器前端生成具备竞争力的代码，被迫全盘转向汇编开发。
2. **对目标 LPU 架构与上层大模型生态的影响**:
   - **开发与维护成本壁垒**: 手写 VLIW 汇编（单文件近 500 KB）极度依赖资深工程师对硬件微架构（发射槽约束、流水线延迟、Bank/Cluster 冲突规则）的精细掌控，开发周期长、调试极其困难；
   - **模型泛化阻碍**: 大模型领域的算子演进极快（如从标准 Attention 到 FlashAttention-2/3、MLA、Sliding Window Attention 等），如果每一种变体都需要耗费数月重写手工汇编，将导致上层框架（如 vLLM、Triton、PyTorch）的适配滞后；
   - **结论建议**: 提升 `scale-cc` 编译器的向量软件流水与寄存器调度优化能力，或提供类似 Triton/LLVM 的专用算子编译中转层，是目标 LPU 架构迈向成熟规模化商用的核心生命线。
