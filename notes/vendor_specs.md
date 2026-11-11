# 目标 LPU 芯片硬件规格与参数推导备忘录 (vendor_specs.md)

本文档系统归档芯片厂商技术问询书面记录、厂商官方算子库代码硬编码参数（`device_info.h` 与 `matmul_perf_info.h`）、代数推导指标，以及当前仍待确认的缺失信息与参数矛盾。

---

## §1 厂商技术物料与初期书面答复（2026-11-03 原文记录）

> **信息来源**：2026-11-03 芯片厂商针对技术问询的书面回复原文

1. **官方算子库 / 高性能算子写法**：
   > “有算子库，但 c 的性能比微码差很多，具体数据待评估或者后续反馈”
2. **标称主频 / 时钟域（初期答复）**：
   > “主频 1.425GHz，SPU/MPU 都是一样的，DDR 根据 IP 的主频定的”
3. **峰值算力 / MPU 发射宽度**：
   > “（对 MPU 每周期能发射几条 v16fp32 FMA）8 条”
   > *注：单 APE / 单 APC / 整芯片的 TFLOPS 数字，以及 FP16 / BF16 / INT8 的情况，厂商未书面回答。*
4. **片外存储规格与带宽（初期答复）**：
   > “GDDR6，单芯片一共支持 8 路 GDDR，2 通道，每通道 16bit@16Gbps”
5. **片间互联**：
   > “互联方式不同芯片不同，8 卡双 full mesh 结构，带宽整卡 1024GB/s，支持 P2P，硬件不支持集合通信，软硬协同支持。”
6. **三类数据区分**：
   > “真实的芯片规格如上问题反馈；CModel 的配置参数和有效性能后续反馈，会尽快协调”

---

## §2 由初期书面答复推导的计算与访存指标（留档备查）

> [!NOTE]
> **说明**：以下指标基于 §1 初期书面回复经明确公式代数推导得出，标注为**初期推导值，非厂商直接标称**。

1. **单 APE 理论峰值算力（FP32）**：
   - 公式：$$\text{单 APE 算力} = 8 \text{ 条/cycle} \times 16 \text{ lane} \times 2 \text{ FLOP} = \mathbf{256 \text{ FLOPs/cycle}}$$
   - 按 1.425 GHz 折算：$$256 \times 1.425 \times 10^9 = \mathbf{364.8 \text{ GFLOPS}}$$
2. **CModel 4 APC (16 APE) 算力**：
   - $$16 \times 364.8 \text{ GFLOPS} = \mathbf{5.837 \text{ TFLOPS}}$$
3. **8 路 GDDR6 物理峰值带宽**：
   - $$8 \text{ 路} \times 2 \text{ 通道} \times 16 \text{ bit} \times 16 \text{ Gbps} / 8 = \mathbf{512 \text{ GB/s}}$$
4. **拐点与带宽分配口径**：
   - **均摊口径 (Amortized)**：整芯片 512 GB/s 均摊至 16 APE 为 32 GB/s/APE，理论平衡拐点 $R = 364.8 / 32 = \mathbf{11.40 \text{ FLOP/byte}}$（具有尺度不变性）；
   - **独占口径 (Exclusive)**：单 APE 独占 512 GB/s，拐点 $R = 364.8 / 512 = \mathbf{0.71 \text{ FLOP/byte}}$（单核空载理论乐观上界）。

---

## §3 多核扩展实测依据与带宽假设检验 (T1 实测结论)

在同一构建产物（`build_id: bbf153e9b289b92bcf8055794e812953`，`results/gemm_multicore.csv`）下，对单 APC 内部 1、2、4 线程进行了扩展性测试：
- **硬件执行周期与加速比**：
  - $M=1$：1T（648,903 cycles）$\to$ 2T（450,417，1.44×）$\to$ 4T（351,151，**1.85×**）；
  - $M=8$：1T（3,775,628 cycles）$\to$ 2T（2,183,843，1.73×）$\to$ 4T（1,387,888，**2.72×**），整核吞吐率达到 0.7555 FLOPs/cycle；
  - 4 线程下 N 模式与 C 模式硬件周期仅差 4 个周期（1,387,888 vs 1,387,892），性能未观察到显著差异。
- **访存带宽扩展性客观定性**：
  实测 4 线程聚合带宽为 0.76 字节/周期（折合 ~1.08 GB/s @ 1.425GHz 或 ~1.22 GB/s @ 1.6GHz），仅占整卡物理带宽的 0.04%（或初期 512 GB/s 的 0.2%），系统访存处于极度轻载状态。
  因此，聚合带宽随线程数近似线性增长仅反映在当前编译器与 DMA 发起水平下片外访存带宽尚未成为多核争用瓶颈，**不能直接判定独占口径成立**。体系结构建模中应以均摊口径为整机基准，独占口径仅作为单核空载时的理论上界。

---

## §4 官方代码库常量 (Vendor Codebase Constants)

以下数值来自厂商官方算子库 Demo 代码库中的硬编码常量，**属于厂商给出的代码级标称值，不是推导值**：

### 1. 拓扑与主频常量
**出处：`mablaslt/mablasLt/common/device_info.h`**
```cpp
k_die_per_card        = 2;
k_apc_num_per_die     = 32;
k_ape_num_per_apc     = 4;
k_spu_main_frequency  = 1.6e9;        // 1.6 GHz
k_max_dm_num          = 24;           // 24 Banks per APC
k_max_dm_byte_size    = 256 * 1024;   // 256 KB per Bank
```
- **由此确立的拓扑层次**：
  - 单 Die 规格：$$32 \text{ APC} \times 4 \text{ APE} = \mathbf{128 \text{ APE}}$$
  - 物理整卡规格：$$2 \text{ Die} \times 128 \text{ APE} = \mathbf{256 \text{ APE (64 APC)}}$$

### 2. 算力与带宽天花板常量
**出处：`mablaslt/mablasLt/common/matmul_perf_info.h`**
```cpp
k_max_memory_bandWidth_gbs          = 3.2 * 1024;   // 3276.8 GB/s
k_max_8bit_compute_bandWidth_gflops = 840'000;      // 840 TFLOPS  (INT8 / FP8)
k_max_16bit_compute_bandWidth_gflops= 420'000;      // 420 TFLOPS  (FP16 / BF16)
k_max_32bit_compute_bandWidth_gflops=  52'500;      // 52.5 TFLOPS (FP32)
```

### 3. 数学一致性验算与重要分析
1. **单 Die 与峰值算力关系验算**：
   $$128 \text{ APE} \times 256 \text{ FLOPs/cycle} \times 1.6 \text{ GHz} = 52,428.8 \text{ GFLOPS} = 52.4288 \text{ TFLOPS} \approx \mathbf{52.5 \text{ TFLOPS}}$$
   由此直接证明：**代码库中的 52.5 TFLOPS 为单 Die (128 APE) 的标称算力**，而非物理双 Die 整卡。
   > [!NOTE]
   > **取整常数脚注**：代码库头文件中定义的 `52'500` (52.5 TFLOPS) 为基于 52.4288 TFLOPS 的名义取整值，差异约 0.136%（~0.14%），不影响任何定性与定量分析结论；其 16-bit 常量 `420'000` (420 TFLOPS) 与 8-bit 常量 `840'000` (840 TFLOPS) 严格为其整 8 倍与整 16 倍。单 APE 理论值严格遵循微架构推导值 409.6 GFLOPS (256 FLOPs/cyc @ 1.6 GHz)。
2. **精度倍率与硬件单元**：
   - FP16/BF16 峰值（420 TFLOPS）为 FP32 的 **8 倍**；
   - INT8/FP8 峰值（840 TFLOPS）为 FP32 的 **16 倍**；
   - 这表明目标 LPU MPU 内部包含专用高吞吐低精度矩阵乘累加硬件单元（非通用 SIMD 简单翻倍）。
3. **单 APE 算力与 878× 差距的不变性**：
   - 单 APE 的 FP32 峰值仍为：$$8 \text{ 条/cycle} \times 16 \text{ lane} \times 2 \text{ FLOP} = \mathbf{256 \text{ FLOPs/cycle}}$$；
   - 此前项目基于实测得出的「单 APE 实测 0.2915 FLOPs/cycle 占峰值的 0.114%，差距 878×」这一结论**建立在每周期算力 (FLOPs/cycle) 口径上，完全不受主频由 1.425 GHz 调整至 1.6 GHz 的影响**。878× 的三层差距分解（8× 发射槽 × 74× 编译器未流水 × 1.48× 控制开销）完全保持成立。

---

## §5 两套参数体系对比与物理时间换算规范

### 1. 两套参数体系对比表

| 硬件指标项 | 初期问询答复 (2026-11-03) | 官方代码库硬编码常量 (`device_info.h` / `matmul_perf_info.h`) | 差异倍数与性质分析 |
| :--- | :---: | :---: | :---: |
| **标称主频** | 1.425 GHz | **1.6 GHz** | +12.3% (可能为标准频 vs 升频/不同批次) |
| **全卡 APC 规模** | 未答 (CModel 默认 4) | **32 / Die (整卡 64 APC / 256 APE)** | 物理整卡为 CModel 默认实例的 16 倍 |
| **片外内存物理带宽** | GDDR6 512 GB/s | **3276.8 GB/s** | 差 6.4 倍 (3276.8 GB/s 达 HBM 级别，可能为下一代/不同形态产品) |
| **FP32 理论峰值 (单 Die)** | 5.837 TFLOPS (按 16 APE 算) | **52.5 TFLOPS (128 APE)\*** | 与 128 APE 验算高度一致 (52.4288 TFLOPS) |
| **FP16 理论峰值** | 未知 | **420 TFLOPS\*** | 8× FP32 (整 8 倍名义值) |
| **INT8 理论峰值** | 未知 | **840 TFLOPS\*** | 16× FP32 (整 16 倍名义值) |

*\*注：单 Die 标称 52.5 TFLOPS 为 52.4288 TFLOPS ($128 \times 256 \times 1.6\text{ GHz}$) 的取整值，差异约 0.14%，不影响分析结论；420 TFLOPS 与 840 TFLOPS 严格为其整 8 / 16 倍。*

### 2. 物理时间换算规范说明
由于厂商书面回复与代码库硬编码主频存在 1.425 GHz 与 1.6 GHz 的分歧，在向厂商最终书面澄清前，本项目采取**并列呈现两组物理时间换算**的规范：
- 若以 1.425 GHz 计算：$$1 \text{ cycle} = \frac{1}{1.425 \times 10^9 \text{ Hz}} \approx \mathbf{0.7018 \text{ ns}}$$；
- 若以 1.6 GHz 计算：$$1 \text{ cycle} = \frac{1}{1.6 \times 10^9 \text{ Hz}} = \mathbf{0.6250 \text{ ns}}$$。
**理由**：两套数值均具备明确厂商出处，在未获官方正式 Datasheet 确认前，并列注明可确保分析严谨性，且核心微基准指标（FLOPs/cycle、Speedup、SRAM 局部性增益）均直接基于硬件周期，不受主频换算影响。报告凡涉及绝对物理时延或峰值带宽处，均标注“参数存在两套来源，结论可能随澄清而变”。
