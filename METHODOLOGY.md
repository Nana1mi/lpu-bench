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
   - 相同工作量下，Device `tick()` 读出的硬件周期数在 3 次重复中**完全一致**（如 10 循环恒为 14,752 cycles，80 循环恒为 117,582 cycles），方差为 0。证明 gem5 仿真核指令周期计算是精确、可重复的硬件级建模。
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
