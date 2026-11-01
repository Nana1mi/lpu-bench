# 工作进度日志 (PROGRESS.md)

## 2026-10-31

### 1. 环境验证与基线确认 (已完成)
- **宿主机**：Windows 11 WSL2 (Ubuntu 24.04)，开启 KVM 嵌套虚拟化与 Linger 常驻。
- **客体机 (QEMU CModel)**：通过 tmux 保持后台运行，Gem5 4 APC 仿真核心就绪，A76 固件就绪。
- **端口与连通性**：SSH 端口自动提取成功，宿主机通过 `scripts/guest_run.sh` 免密直连。
- **健康检查**：
  - `modprobe maps` 成功加载内核模块。
  - `/dev/dri/card1` 设备节点正常。
  - `umd_test/memory/memory_basic_test` 8 项测试全部 PASS。
- **官方 Demo 验证**：
  - 成功编译并执行 `umd_test/kernels` 完整 demo 工程（`a.out`），涵盖 `vaddu32`、`vmulu32`、`vsubu32`、`vtanhfp32` 等 kernel，计算结果校验全部通过（Test PASS）。

### 2. 阶段 0：工程解析与 MPU 特性梳理 (已完成)
- **产出**：
  - `notes/demo_walkthrough.md`：详细解析了 host 与 device 编译流程（`-Xarch_host`, `-Xarch_device`, `--scale-link`）、物理地址转换（`mapsMemobjGetPhysicalAddress`）、SPU 与 MPU 协作流。
  - `notes/mpu_capabilities.md`：梳理了 512-bit 向量类型（`v16fp32`, `v32fp16`, `v64i8` 等）、SPU 调 MPU 机制（`async_call`, `sync_call`, `async_enqueue`）及向量指令与 FMA 累加器。

### 3. 阶段 1：计时方法学建立与基准实验验证 (已完成)
- **统一计时抽象**：完成 `common/timer.h`，确立以 SPU 硬件寄存器 `0x803440` 的 `tick()` 周期计数为唯一基准指标（`mapsEventElapsedTimeEx` 经核验已排除）。
- **微基准测试**：在 `kernels/test_timer/` 下设计了 1x (10), 2x (20), 4x (40), 8x (80) 工作量、各 3 次重复测量的完整套件。
- **关键实验发现**：
  - 测得原始数据写入 `results/timing_validation.csv`。
  - **方差为 0**：在相同配置下，Device 硬件 cycle 重复 3 次完全恒定（10 循环恒为 14,752 cycles，80 循环恒为 117,582 cycles）。
  - **完美线性度**：硬件 cycle 随计算量严格线性增长（$Cycles = 1469 \times \text{loop\_count} + 62$），判定系数 $R^2 = 1.00000$。
  - **仿真时间可信性**：Host 事件时间与 Device cycle 均真实反映 gem5 内部仿真进度，与宿主机墙钟解耦。
- **产出**：完成规范性报告 `METHODOLOGY.md`。

### 4. 阶段 2：三类 LLM 典型算子实现与基准评测 (已完成)
- **K1: FFN / GEMM 扩展性评测 (`kernels/gemm/`)**
  - 实现标准单精度分块矩阵乘算子，针对 $K=256, N=256$ 探索 $M \in \{1, 2, 4, 8\}$ 的扩展区间。
  - 实现 `v0_serial`（严格串行）与 `v1_overlap`（全双缓冲与 `async_enqueue` 硬件队列），每次 repeat 均真实通过 CPU 双精度数学参考模型校验（记录实际 `max_diff`）。
  - 测得原始数据写入 `results/gemm.csv`（24 条真实记录）：
    - 吞吐率随 $M$ 稳步爬升：0.2786 ($M=1$) $\to$ 0.2858 ($M=2$) $\to$ 0.2896 ($M=4$) $\to$ 0.2915 FLOPs/cycle ($M=8$)。
    - 在优化后的全双缓冲与 `async_enqueue` 实现下，流水线固定调度开销大幅收敛至约 1,900~2,000 周期。在 $M=1$ 极小规模下仍存在微量控制开销导致轻微负收益 (0.9960x)，随 $M$ 增大开销被完全稀释。
- **K2: LLM Decode Attention 访存局部性评测 (`kernels/decode_attn/`)**
  - 实现单查询解码注意力算子，设计了基于 IEEE-754 位操作的 Cody-Waite 快速指数 `fast_expf` 与片上归约 Softmax（宿主机实测最大相对误差 $5.57 \times 10^{-5}$）。
  - 实现 `v0_ddr`、`v1_dm_resident`（片上 SRAM 常驻）、`v2_overlap`（双缓冲流式预取）。
  - 测得原始数据写入 `results/decode_attn.csv`（30 条真实记录）：
    - 注意力耗时随序列长度 $L \in \{64, 128, 256, 512\}$ 呈现极高精度线性外推：$Cycles(L) = 2276.5 \times L + 3484.5$ ($R^2 = 0.99999$)。
    - 确定性量化了片上 SRAM 常驻收益：常驻带来 1.082x 的加速，节省了每 Token 约 176.8 cycles 的纯 DDR DMA 搬运时延。
    - 实测明确了单 Bank 256 KB 的物理容量上限，在单 APE 下支持高达 $L=1024$ 的 KV Cache 片上常驻（线性拟合优度 $R^2=1.00000$）。当 $L > 1024$ 时触发 JIT 运行时链接报错（`exceeds max DM size 256KB`）。

### 5. 阶段 3：架构消融实验与可视化分析 (已完成)
- 完成自动消融统计框架 `scripts/run_k3_ablation.py`，生成 `results/ablation.csv`。
- 完成绘图脚本 `scripts/plot.py`，全自动化生成 4 张高清图表：
  - `figures/k1_gemm_utilization.png`：GEMM 有效吞吐率与周期数随 $M$ 变化曲线
  - `figures/k2_attn_cycles.png`：Decode Attention 执行周期随 $L$ 扩展曲线
  - `figures/k2_breakdown.png`：Decode Attention 计算与搬运时间占比堆叠图
  - `figures/k3_speedup.png`：K3 算子流水双缓冲与片上 SRAM 局部性加速比柱状图对比

### 6. 阶段 4：毕设论证报告与一键复现工程 (已完成)
- 完成综合性学术评估报告 `REPORT.md`，基于实测数据深入论证了：
  - RQ1: AF 分离定位（在单 APE 口径下，Aurora 对无状态 FFN 分块计算具备更好的架构适应性，而在长上下文 Attention 下受制于单 Bank 256KB 容量与外部访存开销）
  - RQ2: PD 分离定位（Aurora 在单 APE 口径下更倾向于承担中等 Batch 的 Prefill / 微批解码，而在 Batch=1 的单 Token 解码中难以获得流水重叠收益）
  - RQ3: LPU 设计空间定位（对比 Groq、Cerebras、Tenstorrent、TPU、GPU，阐明 Aurora 的相对优势与演进方向）
  - 针对下一代 Aurora 芯片提出三项硬件演进建议（提供跨 Bank 聚合机制扩充连续 SRAM、升级 HBM/LPDDR5X、引入硬件自动异步流水控流并扩大向量寄存器堆）。
- 实现一键端到端全自动复现脚本 `scripts/reproduce_all.sh`。

---

## 终态交付物核对 (Definition of Done)
- [x] 阶段 0：demo 能复现运行；`notes/demo_walkthrough.md` 和 `notes/mpu_capabilities.md` 完成
- [x] 阶段 1：`METHODOLOGY.md` 写明经过验证的仿真时间计时方法
- [x] K1 GEMM：v0、v1 两个版本正确性全部通过，完成 M 扫描
- [x] K2 decode attention：v0、v1、v2 三个版本正确性全部通过，完成 L 扫描
- [x] K3：加速比汇总完成 (`results/ablation.csv`)
- [x] `results/` 中有完整的原始 CSV；`figures/` 中有 4 张图，全部由脚本生成
- [x] `REPORT.md` 完成，第 5 节明确回答核心问题，所有数字均能追溯到 CSV
- [x] 存在一键复现脚本 (`scripts/reproduce_all.sh`)
- [ ] `BLOCKERS.md`：归档 5 项芯片规格参数缺失与硬件/编译器层面的未解决项（芯片主频/标称算力/DDR理论带宽未知、多 Bank 聚合机制未知、驱动计时 API 语义异常、编译器未开动态流水、Attention d=128 向量寄存器溢出），均已在 REPORT.md 第 6 节（局限性）详细论述。
