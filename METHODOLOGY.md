# 性能测量方法学与仿真时间基准验证报告 (METHODOLOGY.md)

## 1. 核心结论与计时源选型

在思朗科技（SmartLogicTech）Aurora 芯片的 CModel（QEMU + gem5）仿真环境中，**禁止使用宿主机物理墙钟时间**作为评测依据。经过对官方驱动、工具链底层寄存器及微基准测试的严格实验验证，本项目确立了**以 Device 硬件周期计数器 (`tick()`) 为唯一可信性能度量**的评测体系（`mapsEventElapsedTimeEx` 经核验已排除）：

| 计时源 | 采集机制 | 作用域与测量区间 | 精度与方差 | 用途 |
|---|---|---|---|---|
| **Device 硬件周期计数器 (`tick()`)** | 直接读取 SPU 硬件寄存器 `0x803440` (64-bit cycle counter) | **微秒/周期级精确区间**：可分别或联合测量 DMA 搬入、MPU 核心计算、DMA 写回的纯净硬件周期数（排除了 Host 驱动下发和队列调度抖动） | 方差为 **0**（完全确定性，线性度 $R^2 = 1.0000$） | **主要指标**：算子纯计算/流水线周期数、加速比计算、Roofline 模型构建 |
| **Host 仿真事件 API (`mapsEventElapsedTimeEx`)** | 基于 MAPS 异步 Stream 插入的 Event 戳记 | **Kernel 级端到端区间**：覆盖队列下发、SPU 引导、Kernel 执行到完成同步 | 与宿主机物理墙钟高度线性绑定；`sim_ms` 与 `sim_cycles` 恒等 | **已排除** (仅供对照排查留档，不作为性能指标) |
| **Host 物理墙钟 (`std::chrono`)** | x86 宿主机物理时钟 | 仅作为实验等待耗时记录 | 随宿主机负载和 QEMU/gem5 模拟比率波动 | **仅供实验耗时参考，不计入学术报告指标** |

---

## 2. 验证实验与证据 (Validation Evidence)

我们在 `kernels/test_timer/` 下构建了严密的微基准测试套件，在 1x (10 loops)、2x (20 loops)、4x (40 loops)、8x (80 loops) 向量计算工作量下，各重复运行 3 次，测得原始仿真数据（见 `results/timing_validation.csv`）：

### 2.1 原始测量数据
```csv
workload_mult,loop_count,repeat,sim_ms,sim_cycles,device_cycles,wall_ms
1,10,0,856.0000,856,14752,929.24
1,10,1,814.0000,814,14752,882.55
1,10,2,810.0000,810,14752,879.02
2,20,0,1303.0000,1303,29442,1381.16
2,20,1,1399.0000,1399,29442,1480.76
2,20,2,1371.0000,1371,29442,1446.23
4,40,0,2497.0000,2497,58822,2569.10
4,40,1,2408.0000,2408,58822,2488.50
4,40,2,2399.0000,2399,58822,2467.45
8,80,0,4548.0000,4548,117582,4631.56
8,80,1,4659.0000,4659,117582,4731.68
8,80,2,4694.0000,4694,117582,4765.56
```

### 2.2 Device 周期计数器 (`tick()`) 验证分析

1. **确定性与方差 (Repeatability)**：
   - 相同工作量下，Device `tick()` 读出的硬件周期数在 3 次重复中**完全一致**（如 10 循环恒为 14,752 cycles，80 循环恒为 117,582 cycles），方差为 0。证明 gem5 仿真器在给定的静态指令序列下具有完全确定性与可重复性。
   - **重要限定**：“方差为 0”仅在同一二进制内成立，不同 commit / 不同 build 之间的 cycles 不可直接比对（例如代码微小改动可能引起栈偏移、寄存器重排或指令包对齐变动，导致跨二进制出现微小周期漂移）。跨版本比较加速比时必须保证基线与优化版本在同一构建环境下生成。
   - **二进制身份可核验机制 (`build_id`)**：由于典型研发流程为“修改源码 $\to$ 编译 $\to$ 运行测试 $\to$ git commit 提交”，运行时获取的 `git rev-parse HEAD` 仅反映上一个已提交 commit，而非当前执行二进制的真实状态（例如在尚未提交新代码时运行，HEAD 仍停留在旧 commit）。为彻底杜绝“跨 commit 算加速比”的核验漏洞，我们在测试驱动 (`run_k1_gemm.py`、`run_k2_attn.py`) 中增加了 `build_id` 字段，直接计算被执行 ELF 文件的 MD5 哈希（如 `decode_attn.out` 为 `b3887730448dfc2364cbfb112344f78f`，`gemm.out` 为 `7aa34dfcf47bb4d06513c651f980a002`），与 `commit_hash` 并存，使“同一 build 测量”成为数据中可机器核验的事实。
   - **非敏感分支的周期恒定性旁证**：当我们在 `decode_attn.sc` 中将 `RESIDENT_L` 从 128 扩展至 1024 并重新编译后，重跑 $v0$（DDR 流式基线）和 $v2$（双缓冲重叠）在 $L=256$ 与 $L=512$ 下的 Device cycles 逐位不变（$v0@256=589,837$、$v2@256=593,914$、$v0@512=1,176,205$、$v2@512=1,184,458$）。这是因为 $v0$ 与 $v2$ 仅使用局部 tile 缓冲，未在 DM SRAM 上分配 `K_resident` / `V_resident`，对 `RESIDENT_L` 的改动完全不敏感。这一经验证据为 CModel 仿真器在非变更代码路径上的绝对稳定性提供了极强的经验旁证。
   - **测量日志完整性与 Schema 迁移规范 (Append-Only Measurement Integrity)**：实验 CSV 是严格的时间序列物理测量记录，而非报表文档。当引入 `build_id` 等字段进行 schema 迁移时，历史未采样行严格留空（留白），严禁人为回填后续构建的哈希；任何跨版本对比所需配置，必须经由新 runner 实测生成真实数据行（追加写入）。下游分析与作图脚本（`run_k3_ablation.py`、`plot.py`）只过滤与消费具备有效 `build_id` 的实测行。这一原则确保了全链路数据的物理溯源性与可核验性，杜绝以文档化修饰替代物理测量的隐蔽失真。
2. **完美线性度 (Linearity)**：
   - 基础循环基线（10 循环）= 14,752 cycles。
   - 每增加 10 次循环计算，增量周期为：
     $$\Delta = 29442 - 14752 = 14690 \text{ cycles (即 } 1469 \text{ cycles/loop)}$$
   - 拟合回归方程为：$\text{Cycles} = 1469 \times \text{loop\_count} + 62$（其中 62 cycles 为 SPU 内核启动与基础调用开销）。
   - 预测 40 循环：$14752 + 30 \times 1469 = 58822$ cycles（实测精确为 58,822）。
   - 预测 80 循环：$14752 + 70 \times 1469 = 117582$ cycles（实测精确为 117,582）。
   - 判定系数 $R^2 = 1.00000$！

### 2.3 `mapsEventElapsedTimeEx` 的排除理由（阴性验证分析）

在初步方案中，`mapsEventElapsedTimeEx` 曾被考虑作为 Host 端流式计时源。然而，经过对全套基准测试生成的全部 66 行原始 CSV 数据进行逐行交叉验证，发现该接口存在根本性方法学缺陷，必须予以彻底排除：

1. **数值无量纲化与字段恒等**：
   在所有实测 CSV 中，`sim_ms` 与 `sim_cycles` 的数值**完全相等**（例如 `856.0000` 与 `856`，Attention 算子中均为 `3950.0000` 与 `3950`）。两个理论上量纲完全不同（毫秒时间 vs. 时钟周期数）的字段输出同一数值，结合底层日志中驱动打印的 `imapsEventElapsedTimeEx: Returned mapsSuccess : Elapsed Time = , 856, Elapsed Cycle = , 856`（脱敏实测日志片段已归档提交于 `results/maps_event_driver_log_excerpt.txt`，供可重现查验），证实驱动层并未真正读取 gem5 周期模型，返回值缺乏明确的物理量纲。
2. **宿主机物理墙钟的常数偏移伪装**：
   计算 `wall_ms - sim_ms` 的差值：在 `timing_validation.csv` 中，该差值高度稳定在 68 ~ 84 ms 区间；在 `gemm.csv` 中稳定在 53 ~ 94 ms 区间。这一恒定差值本质上是宿主机端 `std::chrono` 统计范围额外包含的内存搬移与同步调用固定开销。换言之，`sim_ms` 实际上直接反映的是宿主机 x86 物理执行耗时，而非 Aurora 芯片的仿真时钟。这完全命中了 `GOAL.md` 阶段 1 的判定准则（"确认候选计时源反映的是仿真时间而不是宿主机时间。如果两者高度相关、比例恒定，就要怀疑它测的是宿主机时间"）。
3. **`device_cycles` 硬件独立性的旁证**：
   与 `sim_ms` 随宿主机负载和仿真比率波动不同，`device_cycles / sim_ms` 的比值呈现明显的非恒定变化（如 1x 负载下为 $14752 / 830 \approx 17.8$，而 8x 负载下为 $117582 / 4634 \approx 25.4$）。这有力证明了 `device_cycles` 来自真正的片上 SPU 周期脉冲计数器，其递增完全取决于仿真内核所执行的指令周期，与宿主机处理器的瞬时执行速度和系统调度彻底解耦。

**结论**：`device_cycles` 是 CModel 仿真环境下唯一具备物理真实性与可重复性的性能指标。`sim_ms` 与 `sim_cycles` 仅作为排查和溯源证据保留在原始 CSV 中，已从 `REPORT.md` 的所有表格和论述中彻底剔除。

### 2.4 时钟域一致性与物理时间换算基准 (Clock Domain Alignment & Physical Time)

2026-11-03 厂商（思朗科技）书面回复明确声明：“**主频 1.425GHz，SPU/MPU 都是一样的，DDR 根据 IP 的主频定的**”。这一结论确立了底层计时到物理时间映射的关键方法学依据：

1. **SPU 与 MPU 时钟域 1:1 等价性**：
   本项目所有待测 Kernel 中的 `tick()` 计数器均直接读取 SPU 物理寄存器 `0x803440`。此前由于硬件时钟树文档未公开，存在“SPU 控制核与 MPU 向量计算核是否存在分频或倍频关系”的疑问。现厂商确认两者同频（1.425 GHz）同域，因此**无需任何分频换算系数，所有实测得到的 `device_cycles` 直接严格适用于 MPU 向量核周期**。
2. **时钟周期与物理时间换算常数**：
   - 单周期物理时长：$$1 \text{ cycle} = \frac{1}{1.425 \times 10^9 \text{ Hz}} \approx \mathbf{0.701754 \text{ ns}} \approx \mathbf{0.7018 \text{ ns}}$$（推导值，非厂商直接标称）
   - 物理时间折算公式：$$\text{Time (ms)} = \frac{\text{device\_cycles} \times 0.701754}{10^6}$$
3. **访存带宽换算适用边界限定**：
   厂商声明“DDR 根据 IP 的主频定的”，表明板载 GDDR6 物理接口时钟由专用内存控制器 IP 独立驱动，与片上处理器核时钟域解耦。因此：**测试套件中输出的“字节/周期 (bytes/cycle)”，其时间基准严格为 SPU/MPU 处理核时钟周期**。当且仅当将“周期”视为核时钟步长时，其向物理带宽的折算为：
   $$\text{有效带宽 (GB/s)} = \text{effective\_bw\_bytes\_per\_cycle} \times 1.425 \text{ GHz}$$
   严禁将该周期与外部 GDDR6 总线时钟或命令周期混淆。

### 2.5 注意力算子访存量统计与常驻模式口径规范 (Attention Memory Traffic & Intensity Caliber)

在微基准数据表与 `results/decode_attn.csv` 中，K2 Decode Attention 的 `v1`（片上 DM 常驻模式）记录的 `total_bytes = 512` 仅代表设备内核执行及计时窗口内实际发生的 Q（$d \times 4 = 256$ 字节）与 O（$d \times 4 = 256$ 字节）数据搬移。因为在 v1 模式下，全部 $K$ 与 $V$ Cache 已在计时开始前预置在片上 SRAM 中，计时窗口内无 KV 搬运。

**方法学口径规范（硬性限定）**：
1. **计时窗口搬运量 vs. 算法总访存量**：
   - `v1` 的 512 字节仅为**计时窗口内的搬运量**，并非该注意力算子的**算法总访存量**。
   - 完整的算法总访存量必须包含完整的 KV Cache 数据流，口径与 $v0$ 保持完全一致：
     $$\text{Total Algorithmic Bytes} = 2 \times L \times d \times 4 + 512 \text{ 字节}$$
     当隐藏层维度 $d=64$ 时，该值为 $512 \times L + 512$ 字节。
2. **严禁用于计算算术强度或片外带宽**：
   - 严禁将计时窗口内的 512 字节用于计算算子的算法算术强度（Arithmetic Intensity, $I$）或有效片外带宽。若用 512 字节计算，会导致 $I$ 虚高至 $32 \sim 512\text{ FLOP/byte}$，造成“片上常驻使 Attention 变成计算密集算子”的物理假象。
   - 在体系结构分析与 Roofline 模型中，Attention 的算术强度必须按完整算法访存量统一计算：
     $$I_{\text{Attn}} = \frac{\text{FLOPs}}{2 \cdot L \cdot d \cdot 4 + 512} \approx \frac{4 L d}{8 L d} \approx \mathbf{0.50 \text{ FLOP/byte}}$$
     在此统一口径下，$v1$ 与 $v0$ 均精准定位于 $I \approx 0.50\text{ FLOP/byte}$ 的深访存受限区，且 $v1$ 的 y 轴算力吞吐高于 $v0$ 约 8.1% ~ 8.4%，直观反映片上 SRAM 常驻消除 DDR 搬运所带来的性能收益。

---

## 3. 统一计时封装 (`common/timer.h`)

为保证所有待测 Kernel（K1 GEMM、K2 Decode Attention、K3 消融实验）代码的一致性，在 `common/timer.h` 中提供了标准化抽象：

```cpp
#include "maps_runtime_api.h"
#include "tick.h"

// 1. Device 侧：细粒度阶段埋点
unsigned long long t_start = tick();
// [阶段 A: DMA 预取]
unsigned long long t_dma = tick();
// [阶段 B: MPU 向量计算]
unsigned long long t_compute = tick();
// [阶段 C: 写回]
unsigned long long t_end = tick();

// 2. Host 侧：通过 mapsEventElapsedTimeEx 监控 Stream 级仿真时延 (已排除作为基准指标，仅供对照留档)
HostEventTimer timer(stream);
timer.start();
kernel<<<grid, block, 0, stream>>>(...);
timer.stop();
TimingResult result = timer.get_result(dev_cycles);
```

---

## 4. 实验环境运行规范

为消除调试日志导致的严重仿真减速与性能失真，所有算子测试执行时强制配置以下环境变量：
```bash
export MAPS_DUMP_GPU_DEBUG_INFO_ENABLE=false
export MALOG_LEVEL=error
export MAPS_KERNEL_REPEATITION_COUNT=1
```
所有测试脚本均在非交互模式下运行，由 `scripts/guest_run.sh` 自动分发与捕获结果。

---

## 5. CModel 仿真器建模粒度与 MPU 编译开销归因 (T0 定性)

在探索算子性能时，观察到 CModel 上向量 FMA 执行耗时远高于理论单周期吞吐的异常现象（GEMM 中单步向量累加约 95~100 周期）。为解释该异常并定性 CModel 的仿真粒度，我们开展了专题实验与归因分析（详见 `notes/cmodel_mpu_granularity.md`）：

### 5.1 编译器生成质量归因 (反汇编审查)
通过 `clang-offload-bundler` 提取 MPU 二进制并执行 `llvm-objdump -d` 反汇编（见 `notes/gemm_mpu.dis` 与 `kernels/test_fma/test_fma_mpu.dis`），查明高级语言循环存在显著的编译器未优化现象：
1. **乘加未自动融合**：`+= *` 被拆分为独立乘法（`FMA1: 0 + A * B`）与加法（`FMA0: C + temp * 1`），中间插入 5 个静态 NOP 保护气泡。
2. **保守静态调度与 NOP 气泡**：编译器采用全静态冒险保护，每次 LOAD 插入 12 个 NOP，每次 FMA 插入 5 个 NOP。
3. **严重寄存器未分配与栈溢出**：循环计数变量 `i`、基地址指针及局部累加向量在每轮迭代均经历多次全量入栈与出栈（`T15[15] + offset`），单轮累加循环指令包膨胀至 70~80 条。

### 5.2 受控微基准线性验证 (`results/test_fma.csv`)
通过受控微基准对 $N \in \{100, 1000, 10000\}$ 进行扫描，得出精确线性回归方程：
- **Mode 0 (Unfused Reg, `acc += va * vb`)**: $Cycles = 74.00 \times N + 204$ ($R^2 = 1.00000$)
- **Mode 1 (Fused Reg, `acc = acc * va + vb`)**: $Cycles = 69.00 \times N + 217$ ($R^2 = 1.00000$)，比 Mode 0 精确减少 5 周期（即消除拆分乘加之间的 5-NOP 气泡）。
- **Mode 2 (DM Acc / GEMM Pattern, `c[0] += a[0] * b[0]`)**: $Cycles = 87.00 \times N + 146$ ($R^2 = 1.00000$)，包含 DM 向量写回（`BIU2: STORE`）。在真实 GEMM 内核中叠加多重循环步进与标量广播计算后，达到约 95~100 周期/步。
- **Mode 3 (Handwritten VLIW Asm, `official_demo/mpu_add_u32`)**: 在 `mpu_add_u32` 汇编调用外包裹 `tick()`（见 `test_fma` Mode 3，实测 $N \in \{100, 1000, 10000\}$，耗时分别为 436, 1336, 10366 周期，结果写入 `results/test_fma.csv`）。三点最小二乘拟合为 $1.00315 \times N + 334.33$（$R^2 = 0.99999993$），段斜率 100→1000 为 1.0000、1000→10000 为 1.0033，约为 **1.003 周期/op（≈1 周期/op）**。
- **对比定性**：在该模型的记账方式下，手写 VLIW 约 1 周期/op，编译 C 为 74 周期/op，差距归因于工具链未做软件流水。本项目不对物理硅片的能力下任何判断。

### 5.3 仿真器微架构建模粒度审查
审查 gem5 配置文件（`config.ini`）、启动参数与运行日志：
1. **MPU 硬件流水线建模被禁用**：`[system.apc.ape0.mpu]` 中明确声明 `pipeline=false`，MPU 未实现动态微架构流水线仿真。
2. **内存系统为固定时延模型 (`SimpleMemory`)**：工程中完全不存在 `DRAMCtrl`，缺乏对物理 DDR/LPDDR 控制器（Row Buffer、Bank 冲突、刷新与物理时序）的建模。
3. **硬件遥测统计缺失**：`stats.txt` 全程为空（0 字节），缺乏流水线停顿分类和微架构硬件计数器。

### 5.4 仿真环境使用与学术定性规范
基于上述实验证据，对本项目的性能数据做出如下学术定性规范：
1. **撤销“周期精确”表述**：严禁在报告或对外交流中将当前 CModel 称为“周期精确硬件仿真器”（Cycle-Accurate Simulator）。该环境属于**指令集级/功能级仿真器附带粗粒度静态延迟估算模型**。
2. **禁止将测得周期直接等同于物理硬件能力**：CModel 测得的绝对 cycle 周期数与算子耗时，反映的是“**当前早期编译器代码生成质量 + 粗粒度静态时延模型**”下的系统行为，不得将其作为 Aurora 物理量产芯片的算力极限提供给外部评估。

---

## 6. 片上数据存储 (DM) 真实容量上限与边界验证方法

针对先前报告中“32 KB 物理容量上限与 L=128 常驻极限”的表述，本项目深入查阅了《SCALE_Programming_Guide》并实施了参数化容量边界实测，彻底厘清了底层存储机制：

### 6.1 体系结构机理与规范来源
1. **Bank 拓扑与物理规格**：单个 APC 包含 **24 个物理 DM Bank**，每个 Bank 容量为 **256 KB**，单 APC 物理 DM SRAM 总量为 **6 MB**。
2. **APE 分配隔离**：4 个 APE 各自独占 6 个 Bank（APE0: 0~5, APE1: 6~11 等），每个 APE 支持 5 个局部存储区（`__mem0__` ~ `__mem4__`）及 1 个共享存储区（`__shared__`）。
3. **“32 KB 建议”的技术原委**：根据编程指南第 4 章规范，在多 Warp 并发执行时，每个 Warp 均需复制一份 Local Region。为支持最大 8 个 Warp 并发调度（$256\text{ KB} / 32\text{ KB} = 8$），文档强烈建议单个 Local Region $\le 32\text{ KB}$。在微基准测试（单 Warp/单线程）场景下，该并发占有率约束并不限制单个 Region 独占完整 Bank。

### 6.2 容量上限实测与错误边界定位
在 `kernels/decode_attn/decode_attn.sc` 中将常驻缓存参数化为 `RESIDENT_L`，在 `--check-scale-mem` 开启下扫描 $L \in \{128, 160, 192, 224, 256, 320, 384, 512, 768, 1024, 1536, 2048\}$：
- **$L \le 1024$（单区 $\le 256\text{ KB}$）**：离线编译、链接、运行时加载与计算全部正常。验证精度绝对误差严格保持在 $10^{-9}$ 量级，耗时满足严格线性方程 $\text{Cycles} = 2098.82 \times L + 3576$ ($R^2 = 1.00000$)。
- **$L > 1024$（单区 $> 256\text{ KB}$，如 384 KB / 512 KB）**：
  - **离线编译与链接**：离线 `clang++ -c` 与 `clang++ --scale-link` 均正常通过；
  - **运行时错误行为**：在 Host 侧调用 `mapsLaunchKernel` 时，驱动层 `scalehost`（`comgr.cpp`）调用设备链接器 `ld.lld`，精确抛出硬性链接错误：
    ```
    ld.lld: error: section .MEMORY0 size 524288 bytes exceeds max DM size 256KB
    ld.lld: error: section .MEMORY1 size 524288 bytes exceeds max DM size 256KB
    ```
    导致 `mapsLaunchKernel` 返回 `mapsErrorInvalidDeviceFunction`。
- **结论**：单个 Memory Region 的硬件与设备链接器硬性物理上限为 **256 KB**（$d=64$ 时对应 **$L=1024$**），超限时表现为**运行时设备 JIT 链接报错**。


