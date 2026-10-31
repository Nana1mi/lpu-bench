# 性能测量方法学与仿真时间基准验证报告 (METHODOLOGY.md)

## 1. 核心结论与计时源选型

在思朗科技（SmartLogicTech）Aurora 芯片的 CModel（QEMU + gem5）仿真环境中，**禁止使用宿主机物理墙钟时间**作为评测依据。经过对官方驱动、工具链底层寄存器及微基准测试的严格实验验证，本项目确立了**双层协同可信计时体系**：

| 计时源 | 采集机制 | 作用域与测量区间 | 精度与方差 | 用途 |
|---|---|---|---|---|
| **Device 硬件周期计数器 (`tick()`)** | 直接读取 SPU 硬件寄存器 `0x803440` (64-bit cycle counter) | **微秒/周期级精确区间**：可分别或联合测量 DMA 搬入、MPU 核心计算、DMA 写回的纯净硬件周期数（排除了 Host 驱动下发和队列调度抖动） | 方差为 **0**（完全确定性，线性度 $R^2 = 1.0000$） | **主要指标**：算子纯计算/流水线周期数、加速比计算、Roofline 模型构建 |
| **Host 仿真事件 API (`mapsEventElapsedTimeEx`)** | 基于 MAPS 异步 Stream 插入的 Event 戳记 | **Kernel 级端到端区间**：覆盖队列下发、SPU 引导、Kernel 执行到完成同步的整体仿真时间 (ms 与 cycle) | 极高线性度 ($R^2 > 0.999$)，包含微量固定启动开销 | **辅助指标**：整网/端到端 Kernel 调用开销评估 |
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

### 2.2 关键指标分析

1. **确定性与方差 (Repeatability)**：
   - 相同工作量下，Device `tick()` 读出的硬件周期数在 3 次重复中**完全一致**（如 10 循环恒为 14,752 cycles，80 循环恒为 117,582 cycles），方差为 0。证明 gem5 仿真核指令周期计算是精确、可重复的硬件级建模。
2. **完美线性度 (Linearity)**：
   - 基础循环基线（10 循环）= 14,752 cycles。
   - 每增加 10 次循环计算，增量周期为：
     $$\Delta = 29442 - 14752 = 14690 \text{ cycles (即 } 1469 \text{ cycles/loop)}$$
   - 预测 40 循环：$14752 + 30 \times 1469 = 58822$ cycles（实测精确为 58,822）。
   - 预测 80 循环：$14752 + 70 \times 1469 = 117582$ cycles（实测精确为 117,582）。
   - 判定系数 $R^2 = 1.00000$！
3. **仿真时间 vs. 物理墙钟**：
   - 数据表明宿主机消耗的物理墙钟（~4.7 秒）与仿真内部时间处于同量级，但包含宿主机进程调度与仿真器自身的瞬时波动，而设备内部周期数则完全消除了这一噪声。

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

// 2. Host 侧：通过 mapsEventElapsedTimeEx 监控 Stream 级仿真时延
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
