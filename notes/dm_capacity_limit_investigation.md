# 目标 LPU 片上存储 (DM) 真实容量上限与 Bank 分区机制实验分析报告

## 1. 研究背景与任务目标

在前期报告中，曾有论述称“单个局部缓冲区的容量上限被严格限制在 32 KB”、“L=128 是片上常驻极限”，并据此得出了一系列关于硬件容量壁垒和 AF 分离的结论。然而，人工代码审查与深入文档核查指出：
1. `GOAL.md` §1.3 原文指出“强烈建议（N 模式）：单个 memory region $\le 32\text{ KB}$”，属于多 Warp 场景下的编程规范建议，而非物理 Bank 硬件上限。
2. `kernels/decode_attn/decode_attn.sc` 中 $L=128$ 实际源于代码内部的硬编码定义与 host 侧条件检查：
   ```cpp
   __mem0__ v16fp32 K_resident[128 * NUM_VECS];
   if (version == "v1" && L > 128) return 1;
   ```
3. 根据任务书 T7 要求，本项目系统查阅了厂商官方文档《SCALE_Programming_Guide》，并通过参数化微基准在仿真环境下实测从 $L=128$ 一直逐步放大至硬件/链接器报错边界，以确定片上 DM 的物理容量真实上限与超限行为表现。

---

## 2. 文档规范与硬件体系结构机理分析

查阅《SCALE_Programming_Guide客户版.docx》第 3 章（执行模型与存储）与第 4 章（存储区域与检查机制），提炼出以下核心架构事实：

### 2.1 DM Bank 拓扑与分配机制
1. **总 Bank 数量**：单个 APC 包含 **24 个 DM Bank**（P415: "DM分为24个bank"）。
2. **APE 分布**：单个 APC 包含 4 个 APE，各 APE 分配互不冲突的独立 Bank 集合：
   - APE0: Bank 0–5
   - APE1: Bank 6–11
   - APE2: Bank 12–17
   - APE3: Bank 18–23
3. **Region 划分**：每个 APE 可访问 6 个 Region：
   - 5 个 Local Memory Region（修饰符 `__mem0__` ~ `__mem4__`）
   - 1 个 Shared Memory Region（修饰符 `__shared__`，Block 内线程共享）
   - Stack 默认分配在各 APE 的第一个 Bank（APE0 位于 Bank0，APE1 位于 Bank6 等）。
4. **单 Bank 物理容量**：
   - 文档 P757 明确指出：“Bank0 存在 reserved memory，Bank0 可用空间为 256KB - reserved_mem_size，目前 reserved size 为 176KB，可用空间为 80KB。”
   - 硬件单 Bank 的标准容量为 **256 KB**。
   - 整个 APC 的 24 个 Bank 物理 DM SRAM 总量为 $24 \times 256\text{ KB} = 6,144\text{ KB} = \mathbf{6\text{ MB}}$。

### 2.2 为什么编程指南会出现“32 KB 强烈建议”？
根据文档 P801–P802 的“Block size 相关限制”：
$$\text{align}(\max(\text{non\_rmem\_warp\_region\_size})) \times \text{max\_warp\_per\_ape} \le 256\text{ KB}$$
- 当一个 APE 上并发调度多个 Warp 时，非 `__residency__` 属性的局部存储区域（Local Memory Region）需要按 Warp 进行空间复制。
- 如果单个 Region 容量为 32 KB，则单个 APE 最大可支持的 Warp 数为 $256\text{ KB} / 32\text{ KB} = 8\text{ Warps}$。
- 若 Region 进一步增大，可并发的 Warp 数将急剧减少甚至无法调度。因此 32 KB 是为了在多 Warp 并发场景下保障占有率 (Occupancy) 而提出的**工程软建议**。
- 在微基准测试（单 APE、单 Block、单线程/单 Warp，`gridDim(1,1,1)`，`blockDim(1,1,1)`）下，该多 Warp 约束并不生效，单个 Region 理论上可独占单 Bank 的完整容量。

---

## 3. DM 容量上限实测实验设计与结果

### 3.1 实验方法
在 `kernels/decode_attn/decode_attn.sc` 中将常驻容量参数化为 `RESIDENT_L`：
```cpp
__mem0__ v16fp32 K_resident[RESIDENT_L * NUM_VECS] __attribute__((aligned(64)));
__mem1__ v16fp32 V_resident[RESIDENT_L * NUM_VECS] __attribute__((aligned(64)));
```
其中每个 Token 占 $d=64$ 个 `float`（即 $4 \times \text{v16fp32} = 256\text{ 字节}$）。
单区域容量换算公式为：
$$\text{Buffer Size (Bytes)} = \text{RESIDENT\_L} \times 256\text{ 字节} = \frac{\text{RESIDENT\_L}}{4}\text{ KB}$$

使用编译选项 `--check-scale-mem`，对 $L \in \{128, 160, 192, 224, 256, 320, 384, 512, 768, 1024, 1536, 2048\}$ 进行全流程编译、离线链接、运行时加载与正确性验证。

### 3.2 实测数据与执行表现

| RESIDENT_L | 单 Buffer 容量 | 离线编译 | 离线链接 (`--scale-link`) | 运行时加载与执行 | Device Cycles | 精度验证 (Max Diff) | 状态 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **128** | 32 KB | PASS | PASS (warning: sub-target only) | 正常执行 | 272,225 | 6.98e-10 | **PASS** |
| **160** | 40 KB | PASS | PASS | 正常执行 | 339,387 | 1.16e-09 | **PASS** |
| **192** | 48 KB | PASS | PASS | 正常执行 | 406,549 | 2.56e-09 | **PASS** |
| **224** | 56 KB | PASS | PASS | 正常执行 | 473,713 | 1.86e-09 | **PASS** |
| **256** | 64 KB | PASS | PASS | 正常执行 | 540,875 | 1.63e-09 | **PASS** |
| **320** | 80 KB | PASS | PASS | 正常执行 | 675,199 | 1.28e-09 | **PASS** |
| **384** | 96 KB | PASS | PASS | 正常执行 | 809,523 | 1.46e-09 | **PASS** |
| **512** | 128 KB | PASS | PASS | 正常执行 | 1,078,171 | 2.44e-09 | **PASS** |
| **768** | 192 KB | PASS | PASS | 正常执行 | 1,615,469 | 9.60e-10 | **PASS** |
| **1024** | **256 KB** | PASS | PASS | 正常执行 | 2,152,767 | 1.05e-09 | **PASS** |
| **1536** | 384 KB | PASS | PASS | **运行时链接报错** | 0 | 校验失败 | **FAIL** |
| **2048** | 512 KB | PASS | PASS | **运行时链接报错** | 0 | 校验失败 | **FAIL** |

### 3.3 超限报错定位 (Failure Mode)

当 $L > 1024$（例如 $L=1536$ 对应 384 KB，或 $L=2048$ 对应 512 KB）时：
1. **离线编译与链接阶段**：`clang++ -c` 与 `clang++ --scale-link` 均未报错并正常生成 `.out` ELF 可执行文件。
   - 离线链接器打印告警：`ld.lld: warning: --check-scale-mem is only supported on specific sub-target`，说明静态内存检查参数仅在特定子目标下触发，当前目标架构下跳过离线检查。
2. **运行时 JIT 链接阶段**：
   当 host 侧调用 `mapsLaunchKernel` 时，驱动层运行时 `scalehost`（`comgr.cpp`）拉起设备链接器 `ld.lld` 加载链接脚本 `apc-ai.ld`，此时设备链接器精确截断并报错：
   ```
   [2026-11-01 09:57:20.260][error][comgr.cpp(358): ELFLink][scalehost] ld.lld: error: section .MEMORY0 size 524288 bytes exceeds max DM size 256KB
   ld.lld: error: section .MEMORY1 size 524288 bytes exceeds max DM size 256KB
   [2026-11-01 09:57:20.262][error][src/scale_devprogram.cpp(106): build][scalehost] Link failed
   [2026-11-01 09:57:20.263][error][src/scale_global.cpp(64): getStatFunc][scalehost] Build program failed
   [2026-11-01 09:57:20.265][error][src/scale_host.cpp(188): scaleGetKernelInfo][scalehost] getStatFunc return error 1
   [INFO][maps_module.cpp:464] mapsLaunchKernel: Returned mapsErrorInvalidDeviceFunction
   ```
3. **结论**：
   - 单个 DM Memory Region 的**真实物理与链接器硬性上限为 256 KB**。
   - 超出 256 KB 时，错误表现为**运行时设备程序加载失败**（`mapsErrorInvalidDeviceFunction`），而非静态编译失败。

---

## 4. 周期线性关系与性能特征

在 $L \in [128, 1024]$ 这一完全常驻在片上 DM SRAM 的区间内，计算耗时呈现极高的线性度：
- $L = 128 \to 272,225\text{ cycles}$
- $L = 256 \to 540,875\text{ cycles}$ ($\Delta = 268,650$, 斜率 $2098.83$)
- $L = 512 \to 1,078,171\text{ cycles}$ ($\Delta = 537,296$, 斜率 $2098.81$)
- $L = 1024 \to 2,152,767\text{ cycles}$ ($\Delta = 1,074,596$, 斜率 $2098.82$)

严格拟合公式为：
$$\text{Cycles}_{\text{resident}} = \mathbf{2098.82} \times L + 3,576 \quad (R^2 = 1.00000)$$

相较于流式从外部 DDR 访问的 v0 版本（斜率约 $2276.5\text{ cycles/token}$），全常驻模式每 Token 节省约 $177.7\text{ cycles}$ 的 DMA 搬运与同步开销。

---

## 5. 对报告结论的修正要点

1. **推翻“32 KB 硬件物理壁垒”断言**：
   - 32 KB 只是多 Warp 编程规范下的建议单区域分配上限。
   - 单 Region 物理 Bank 上限为 **256 KB**，单 APC 物理 DM 总量为 **6 MB**（24 Banks）。
2. **修正“L=128 极限”断言**：
   - 在隐藏层维度 $d=64$ 时，单 APE 的两个 independent regions（`__mem0__` 与 `__mem1__`）各支持 256 KB，可直接完整常驻 **$L = 1024$ 个 Token** 的 Key 与 Value Cache。
3. **修正报告与建议**：
   - 报告第 1.2 节、第 5.1 节、第 7 节建议 1 中关于“32 KB 硬件容量壁垒”的措辞全部降级与修正。
   - 真实瓶颈并非物理容量只有 32 KB，而是多 Warp 资源划分以及跨 Bank 连续内存聚合机制在当前编程模型中的支持程度。
