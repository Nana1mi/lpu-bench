# 思朗科技 Aurora 芯片硬件规格与参数推导备忘录 (vendor_specs.md)

本文档归档 2026-11-03 思朗科技针对技术问询的书面回复原文、基于标称参数的代数推导指标，以及后续仍待确认的缺失信息。

---

## §1 厂商直接给出的硬件参数（原文照录）

> **信息来源**：2026-11-03 厂商书面回复

1. **官方算子库 / 高性能算子写法**
   > 有算子库，但 c 的性能比微码差很多，具体数据待评估或者后续反馈

2. **标称主频 / 时钟域**
   > 主频 1.425GHz，SPU/MPU 都是一样的，DDR 根据 IP 的主频定的

3. **峰值算力 / MPU 发射宽度**
   > （对"MPU 每周期能发射几条 v16fp32 FMA"）8 条
   >
   > 注：单 APE / 单 APC / 整芯片的 TFLOPS 数字，以及 FP16 / BF16 / INT8 的情况，厂商未回答。

4. **片外存储规格与带宽**
   > GDDR6，单芯片一共支持 8 路 GDDR，2 通道，每通道 16bit@16Gbps

5. **片间互联**
   > 互联方式不同芯片不同，8 卡双 full mesh 结构，带宽整卡 1024GB/s，支持 P2P，硬件不支持集合通信，软硬协同支持。

6. **三类数据区分**
   > 真实的芯片规格如上问题反馈
   > CModel 的配置参数和有效性能后续反馈，会尽快协调

---

## §2 由 §1 标称参数推导的计算与访存指标

> [!IMPORTANT]
> **说明**：以下指标均基于 §1 厂商提供的物理参数经明确公式代数推导得出，标注为**推导值，非厂商直接标称**。厂商未直接给出 TFLOPS 或 GB/s 的总标称值。

1. **单 APE 理论峰值算力（FP32）**：
   - 向量计算规则：MPU 支持 512 位宽向量，单条 `v16fp32` 包含 16 路单精度浮点，单次融合乘加 (FMA) 计为 2 FLOP。
   - 公式：$$\text{单 APE 算力} = 8 \text{ 条/cycle} \times 16 \text{ lane} \times 2 \text{ FLOP} = \mathbf{256 \text{ FLOPs/cycle}}$$（推导值，非厂商标称）
   - 频率折算（主频 $1.425\text{ GHz} = 1.425 \times 10^9\text{ Hz}$）：
     $$\text{单 APE 峰值算力} = 256 \text{ FLOPs/cycle} \times 1.425 \times 10^9 \text{ cycles/s} = \mathbf{364.8 \text{ GFLOPS}}$$（推导值，非厂商标称）

2. **集群与整芯片理论峰值算力（FP32）**：
   - 单 APC 包含 4 个 APE：
     $$\text{单 APC 峰值算力} = 4 \times 364.8 \text{ GFLOPS} = \mathbf{1.4592 \text{ TFLOPS}}$$（推导值，非厂商标称）
   - 整芯片包含 4 个 APC（共 16 个 APE）：
     $$\text{整芯片峰值算力} = 16 \times 364.8 \text{ GFLOPS} = \mathbf{5.8368 \text{ TFLOPS}} \approx \mathbf{5.837 \text{ TFLOPS}}$$（推导值，非厂商标称）

3. **片外存储理论物理峰值带宽**：
   - 存储接口配置：8 路 GDDR6，每路 2 个通道，每通道 16 bit，等效数据传输率为 16 Gbps。
   - 总物理位宽：$$8 \times 2 \times 16 \text{ bit} = \mathbf{256 \text{ bit}}$$
   - 理论物理峰值带宽公式：
     $$\text{峰值带宽} = \frac{256 \text{ bit} \times 16 \text{ Gbps}}{8 \text{ bit/byte}} = \mathbf{512 \text{ GB/s}}$$（推导值，非厂商标称）

4. **硬件时钟周期物理时长**：
   - 公式：$$1 \text{ cycle} = \frac{1}{1.425 \times 10^9 \text{ Hz}} \approx \mathbf{0.701754 \text{ ns}} \approx \mathbf{0.7018 \text{ ns}}$$（推导值，非厂商标称）

5. **Roofline 理论拐点 $R$ (Arithmetic Intensity Turn-Point)**：
   - 拐点公式：$$R = \frac{\text{理论峰值算力}}{\text{理论峰值带宽}}$$
   - **整芯片口径**（16 APE 共享 512 GB/s 带宽）：
     $$R_{\text{chip}} = \frac{5836.8 \text{ GFLOPS}}{512 \text{ GB/s}} = \mathbf{11.40 \text{ FLOP/byte}}$$（推导值，非厂商标称）
   - **单 APC 口径**（4 APE 对应 512 GB/s 带宽）：
     $$R_{\text{apc}} = \frac{1459.2 \text{ GFLOPS}}{512 \text{ GB/s}} = \mathbf{2.85 \text{ FLOP/byte}}$$（推导值，非厂商标称）
   - **单 APE 口径**（单 APE 独立对应 512 GB/s 带宽，微基准口径）：
     $$R_{\text{ape}} = \frac{364.8 \text{ GFLOPS}}{512 \text{ GB/s}} = \mathbf{0.7125 \text{ FLOP/byte}} \approx \mathbf{0.71 \text{ FLOP/byte}}$$（推导值，非厂商标称）

---

## §3 仍未解决与缺失的硬件/仿真信息

以下项目在本次问询中厂商尚未提供，继续作为待跟踪的技术事项：

1. **低精度（FP16 / BF16 / INT8）发射宽度与专用计算单元**：
   - 厂商仅回答了 `v16fp32` 的 8 条发射宽度，未声明半精度（`v32fp16`、`v32bf16`）与低精度整型（`v64i8`）的发射宽度，亦未说明是否存在专用的二维张量/矩阵乘累加硬件单元。
2. **片间互联通信延迟 (Interconnect Latency)**：
   - 厂商已提供 8 卡双 Full Mesh 拓扑与整卡 1024 GB/s 的双向互联带宽，但未提供网络跳步与物理传输延迟。对于延迟极度敏感的单 Token 自回归 Attention 计算，缺乏延迟参数限制了分布式推理端到端时延建模。
3. **硬件集合通信支持机制**：
   - 厂商已明确“硬件不支持集合通信，需软硬协同支持”。这意味着多芯片间的 AllReduce、AllGather、ReduceScatter 需由 SPU 标量核配合软件协议栈调度实现，增加了多卡并行流水线的控制开销。
4. **CModel 仿真器配置参数与有效校准性能**：
   - 厂商说明“CModel 的配置参数和有效性能后续反馈，会尽快协调”。当前 CModel 内部的 MPU 流水线仿真关闭（`pipeline=false`）、DRAM 为固定延迟模型，仍需厂商提供 CModel 的校准数据。
5. **官方高性能算子库（微码/汇编）具体基准性能数据**：
   - 厂商确认“有算子库，但 c 的性能比微码差很多，具体数据待评估或者后续反馈”。官方微码算子库的真实吞吐与利用率尚待获得。
6. **官方标称 TFLOPS 与物理带宽规格文档**：
   - §2 中计算的 5.837 TFLOPS 与 512 GB/s 属于公式推导值，仍需芯片架构团队提供正式的芯片数据手册 (Datasheet) 予以书面确认。
