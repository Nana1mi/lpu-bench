# 阻塞与待解决问题清单 (BLOCKERS.md)

本文档记录本项目在 Aurora 处理器 CModel 仿真评测过程中遇到的阻塞项、未知参数及工具链/驱动层异常。根据 `GOAL.md` 规范，所有无法独立闭环的技术阻碍均在此归档，供后续与思朗科技芯片团队及人类研究员沟通确认。

---

## 1. 芯片关键物理参数缺失（主频、峰值算力、DDR 理论带宽）

- **现象**：
  在开源代码、驱动头文件（`driver/toolchain/include/`）、`SCALE_Programming_Guide`、`docs/` 以及 gem5 配置文件中，均未声明 Aurora 物理处理器的标称主频 (Clock Frequency)、单 APE/单 APC/整芯片的标称峰值算力 (TFLOPs) 与 DDR/LPDDR 物理理论带宽 (GB/s)。
  由于缺失时钟频率，实测周期数（Cycles）无法精确折算为物理墙钟时间（微秒/纳秒），有效访存带宽只能以“字节/周期 (bytes/cycle)”呈现，无法准确计算算力利用率 (Compute Utilization) 与访存利用率 (Bandwidth Utilization)，亦无法定量绘制经典硬件 Roofline 曲线。
- **已尝试的查找方法**：
  1. 检索了驱动与工具链头文件目录下的所有宏定义（如 `AURORA_CLK`, `CLOCK_FREQ`, `DDR_BW` 等）；
  2. 查阅了 `docs/搭建cmodel环境.md` 与 `SCALE_Programming_Guide` 全文；
  3. 检查了 gem5 配置文件 `driver/pciemu/log/gem5/device1/config.ini`，其中内存为 `SimpleMemory`，总线时钟为仿真器默认设定，未见硬件物理标称值。
- **需要人类做什么**：
  向思朗科技架构团队获取 Aurora 芯片的标称工作主频（例如 1.0 GHz / 1.2 GHz 等）、外部存储总线规格（DDR4/DDR5/LPDDR5X 通道数与理论位宽）以及官方硬件标称向量峰值算力（FP32/FP16/INT8 TFLOPs）。

---

## 2. 片上高速数据存储 (DM) 真实物理容量与分区上限机制未知

- **现象**：
  `SCALE_Programming_Guide` 仅提及在单核模式（N 模式）下“强烈建议单个 memory region 容量 $\le 32\text{ KB}$”。当单个缓冲区超过 32 KB 时，编译器报空间溢出错误。然而硬件手册提及片上共有 24 个 DM Bank，且 SCALE 语言定义了 `__shared__`、`__fix0~5__`、`__residency__` 等存储修饰符。目前尚不明确单个 APC 内物理 DM 的真实总容量（例如是 24 × 32KB = 768KB 还是其他规格），亦不明确如何通过编程模型分配超过 32 KB 的连续缓冲区。
- **已尝试的查找方法**：
  查阅了编程指南第 3 章（Memory Space）与第 4 章（DMA Transfer），发现当前驱动默认示例仅使用 `__mem0__` ~ `__mem4__` 分区；在 K2 评测中受单分区 32 KB 限制，仅测定了 $L \le 128$（$128 \times 64 \times 4\text{B} = 32\text{ KB}$）的常驻表现。
- **需要人类做什么**：
  向思朗科技系统软件团队确认：
  1. Aurora 芯片单 APC 与整芯片的物理 DM SRAM 总容量；
  2. 编译器在启用 `--check-scale-mem` 时对各个 Bank 分区的硬性检查规则；
  3. 是否支持跨 Bank 聚合申请大容量常驻缓存（如支持 $L \ge 256$ 上下文常驻）。

---

## 3. `mapsEventElapsedTimeEx` 驱动计时 API 返回值语义不明

- **现象**：
  驱动层提供的硬件事件计时 API `mapsEventElapsedTimeEx(&ms, start, end)` 返回的毫秒数值与 `sim_cycles` 逐行恒等（例如 `ms = 3950.0000`, `cycles = 3950`）。且 `wall_ms - sim_ms` 在不同负载下均恒定在 50~90 ms 之间，表现为宿主机墙钟耗时减去固定的 host 侧 memcpy/同步开销。该指标已被证实反映的是宿主机运行时间而非芯片仿真内部时间，无法作为独立的性能度量。
- **已尝试的查找方法**：
  排查了 `driver/pciemu/log/` 下的驱动日志，观察到 `imapsEventElapsedTimeEx: ... Elapsed Time = , 856, Elapsed Cycle = , 856`。确认驱动底层将同一时间量直接赋给了时间与周期字段。
- **需要人类做什么**：
  向思朗科技软件驱动团队反馈该现象，确认 `mapsEventElapsedTimeEx` 在 CModel 仿真模式下是否仅作为 mock 桩实现，以及驱动是否存在将宿主机 `gettimeofday` 或 `clock_gettime` 直接作为 device cycle 返回的问题。

---

## 4. CModel 仿真建模粒度与编译器向量软件流水调度局限

- **现象**：
  1. 经核查 gem5 配置与运行时日志，CModel 中 MPU 的动态流水线模拟被显式关闭（`pipeline=false`），片外访存使用粗粒度固定延迟模型 `SimpleMemory`，无真实 DRAM 控制器时序与动态流水线 stall 统计。
  2. `scale-cc` 编译器（基于 LLVM 12）存在优化缺陷：未将朴素 `+= *` 融合为硬件 FMA 指令；未能在循环迭代间复用向量寄存器，导致每个迭代将循环感应变量与指针频繁溢出至栈内存（`T15[15]`）；且未实现软件流水，插入了大量静态 NOP 等待气泡。导致实测单条 `v16fp32` 乘加消耗 74~87 周期（而在官方手写汇编 `official_demo` 中通过软件流水单操作仅需 1 周期）。
  3. 当前测得的 cycle 数主要受制于编译器生成质量，无法直接代表物理硅片的执行上限。
- **已尝试的查找方法**：
  对 `kernels/gemm` 与微基准 `kernels/test_fma` 进行了 MPU 汇编解包与反汇编分析（`notes/cmodel_mpu_granularity.md`），建立了单向量指令周期线性回归模型（$R^2 = 1.00000$）。
- **需要人类做什么**：
  向思朗科技编译器团队咨询是否具备开启向量循环展开与软件流水调度的优化开关，或获取官方高性能数学库（BLAS / Attention）的手写汇编实现；向仿真器开发团队咨询未来版本是否计划支持周期精确的 MPU 动态流水线与 DRAM 冲突仿真。

