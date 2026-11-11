# 目标 LPU CModel MPU 建模粒度与编译指令开销深度归因报告 (T0)

**报告日期**: 2026-11-01  
**研究对象**: 目标 LPU MPU (Matrix/Vector Processing Unit) 在 CModel 仿真环境中的时延模型与编译器行为  
**实验工具**: `scale-cc` (Clang/LLVM 12), `clang-offload-bundler`, `llvm-objdump -d`, `gem5.opt` (HPP2MPU), `tick()` (硬件周期计数器)  

---

## 1. 核心异常现象与研究目标 (The Anomaly)

在基于 CModel 的基准测试中，我们观察到一个显著违背向量处理器微架构常理的性能异常：
- **理论微架构预期**：目标 LPU MPU 具备 512-bit 向量执行单元（单时钟周期可执行 1 条 16 路单精度浮点 FMA，理论算力达 16 FLOPs/cycle）。一个单纯的向量累加循环在理想硬件上单步吞吐量应为 **1 周期量级**。
- **CModel 实测现象**：
  - 在 K1 GEMM 算子中，`mpu_gemm_accumulate_tile` 内核实测耗时折合每次 `v16fp32` 累加步耗费 **约 95~100 个周期**。
  - 算子有效计算吞吐率仅为 ~0.29 FLOPs/cycle，距理论 16 FLOPs/cycle 存在接近两个数量级（$\sim 55\times$）的巨大鸿沟。

**研究目标 (T0)**：
通过反汇编二进制分析、受控向量微基准测试（$N=100/1000/10000$ 线性拟合）、以及 CModel 仿真器底层配置源码审查，彻底定性并解释“每条向量 FMA 约 100 周期”的底层根因，评估该仿真模型的建模粒度。

---

## 2. 三维度系统性归因调查

### 2.1 维度 A：MPU 目标汇编反编译审查 (Compiler Code Generation)

使用 `clang-offload-bundler` 从宿主 fat-binary 中剥离出目标为设备端 MPU 的机器码 ELF，并调用 `/opt/maps/toolchain/bin/*/llvm-objdump -d` 获得纯反汇编。完整汇编文件见 `notes/gemm_mpu.dis` 与 `kernels/test_fma/test_fma_mpu.dis`。

反汇编分析揭示了 `scale-cc` 编译器在处理 C++ 向量类型（`v16fp32`）时的三大低效特征：

1. **高级语义未融合为 FMA 指令（乘加拆分）**：
   在 C++ 源码中编写的 `acc += va * vb`，LLVM 12 并未匹配为一条 FMA 硬件指令，而是退化拆解为两条独立指令：
   ```asm
   // 乘法指令：计算 0 + va * vb -> 暂存寄存器 FMA0.T0
   FMA1: 0 + FMA1.T0 * FMA1.T1 (FP32) -> FMA0.T0(MODE1) ;
   // 编译器静态插入 5 个 NOP 气泡以保护数据冒险
   NOP ; NOP ; NOP ; NOP ; NOP ;
   // 加法指令：计算 acc + FMA0.T0 * 1 -> 写回寄存器 BIU.T0
   FMA0: FMA0.T1 + FMA0.T0 * 1 (FP32) -> BIU.T0(MODE1) ;
   ```
   只有显式书写为 `acc = acc * va + vb` 时，编译器前端 AST 才能正确识别并直接生成单条 FMA 指令：
   ```asm
   FMA0: FMA0.T2 + FMA0.T0 * FMA0.T1 (FP32) -> BIU.T0(MODE1) ;
   ```

2. **编译器静态流水调度插入海量 NOP 气泡**：
   目标 LPU MPU 属于 VLIW 静态编译调度架构，硬件不包含复杂的动态 Scoreboard / 硬件联锁（Hazard Interlock）。`scale-cc` 采用保守的静态延迟填充：
   - 每次向量访存（`BIU LOAD`）之后，硬编码插入 **12 个 NOP** 指令。
   - 每次向量乘法（`FMA1`）之后，硬编码插入 **4~5 个 NOP** 指令。
   - 分支跳转前插入 **5~6 个 NOP** 指令。

3. **极端严重的寄存器未分配与栈溢出（Stack Spilling）**：
   `scale-cc` 未能对 MPU 本地循环建立跨迭代的寄存器生命周期分配：
   - 循环迭代计数器 `i`、基地址指针、甚至循环累加的向量中间变量，在**每次迭代**中均经历全量保存到栈 `T15[15] + offset` 并重新从栈加载的过程。
   - 仅单次循环体内部，就包含了 3~4 次独立的 `BIU LOAD` 和 `BIU STORE` 栈操作，每次访存均伴随 12 个静态 NOP。

**静态指令与周期汇总**：
在 `test_fma_mpu.dis` 的循环体 `LBB0_2` + `LBB0_3` + `LBB0_1` 中，单次循环共计包含：
- 4 次向量/标量 BIU 访存 + 48 个访存 NOP
- 2 次 FMA 运算 + 9 个计算 NOP
- 1 次标量加法 + 5 个运算 NOP
- 2 次分支控制 + 6 个控制 NOP
**单轮循环静态指令包累计恰好为 74~77 条指令**，从汇编静态长度即可直接预见每个循环步约需 74 周期。

---

### 2.2 维度 B：受控向量微基准测试与手写汇编对比 (Empirical Slope & Benchmark)

为了精确测量单条向量指令的增量周期，我们构建了纯受控微基准工程 `kernels/test_fma/`，排除外围 DMA 搬运干扰，在设备端使用 `tick()` 测量不同迭代次数 $N \in \{100, 1000, 10000\}$ 的绝对硬件周期（每次配置重复 3 次）。

#### 实验设计：三种模式对比
- **Mode 0 (Unfused Reg)**: `acc += va * vb`（编译为 FMA1 乘 + 5 NOP + FMA0 加）
- **Mode 1 (Fused Reg)**: `acc = acc * va + vb`（编译为单条 FMA0 指令）
- **Mode 2 (DM Mem Acc)**: `c[0] += a[0] * b[0]`（匹配 GEMM 真实访存累加模式）

#### 实测数据记录 (`results/test_fma.csv`)

| 模式 | 迭代量 $N$ | 重复测量 (Rep 1/2/3) | 周期数 (Cycles) | 平均单步周期 | 线性回归拟合方程 | 判定系数 $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode 0 (Unfused Reg)** | 100 | 7604 / 7604 / 7604 | 7,604 | 76.04 | $Cycles = \mathbf{74.00} \times N + 204$ | **1.00000** |
| | 1,000 | 74204 / 74204 / 74204 | 74,204 | 74.20 | | |
| | 10,000 | 740204 / 740204 / 740204 | 740,204 | 74.02 | | |
| **Mode 1 (Fused Reg)** | 100 | 7117 / 7117 / 7117 | 7,117 | 71.17 | $Cycles = \mathbf{69.00} \times N + 217$ | **1.00000** |
| | 1,000 | 69217 / 69217 / 69217 | 69,217 | 69.22 | | |
| | 10,000 | 690217 / 690217 / 690217 | 690,217 | 69.02 | | |
| **Mode 2 (DM Acc / GEMM)**| 100 | 8846 / 8846 / 8846 | 8,846 | 88.46 | $Cycles = \mathbf{87.00} \times N + 146$ | **1.00000** |
| | 1,000 | 87146 / 87146 / 87146 | 87,146 | 87.15 | | |
| | 10,000 | 870146 / 870146 / 870146 | 870,146 | 87.01 | | |

#### 关键拟合结论：
1. **Mode 0 与 Mode 1 的差值精确等于 5 周期**：
   $\Delta Slope = 74.00 - 69.00 = 5.00$ 周期。该 5 周期差值与反汇编中乘法与加法之间插入的 **5 个 NOP 气泡** 完全吻合，证实了 CModel 严格逐指令推进静态发射。
2. **Mode 2 访存写入开销**：
   相比纯寄存器累加，Mode 2 每次迭代增加了一次 DM 向量写回（`BIU2: STORE`），斜率上升为 **87.00 周期**。在完整 GEMM 中，由于存在内层三重嵌套循环计算步进、标量广播地址偏移等开销，单步增量周期进一步上升至 **~95–100 周期**。
3. **方差为 0**：在 CModel 中，无论测量多少次，cycle 数值恒定不变，证明其行为属于确定性静态模拟。

#### 与厂商官方手写汇编的对比
查阅官方 Demo 目录下的手写汇编算子 `kernels/official_demo/mpu_add_u32.m.asm`：
```asm
.global	mpu_add_u32
mpu_add_u32:
    R0:M[0]->BIU.T0 || R1:M[1]->BIU.T4 || R2:M[2]->BIU.T8;
    NOP ; NOP ; NOP ;
    // 关键：在进入循环前对各单元进行软件流水（Software Pipelining）初始化
    BIU0: wait 0 || BIU1: wait 0 || SHU0: wait 13 || BIU2: wait 16;
LOOP:
    // 单行 VLIW 指令包：同时执行 2路Load + 1路向量加法 + 1路Store + 循环计数自减跳转
    BIU0: LOAD(T0)(A++)->SHU0.T0 || BIU1: LOAD(T4)(A++)->SHU0.T1 || SHU0: T0+T1(T)(U)(I32)->BIU.T3 || BIU2: Store(T3,T8)(A++) || MFETCH: IF ( 1 != KI0 ) JUMP %LOOP(DECKI);
```
**对比结论**：
- 在该模型的记账方式下，手写 VLIW 约 1 周期/op，编译 C 为 74 周期/op，差距归因于工具链未做软件流水。
- 经在 `official_demo` 的 `mpu_add_u32` 外包裹 `tick()` 实测（在 `test_fma` Mode 3 下实测 $N \in \{100, 1000, 10000\}$），实测周期分别为 436, 1336, 10366 周期。三点最小二乘拟合为 $1.00315 \times N + 334.33$（$R^2 = 0.99999993$），段斜率 100→1000 为 1.0000、1000→10000 为 1.0033，约为 **1.003 周期/op（≈1 周期/op）**。
- 本测试不对物理硅片的能力下任何判断，仅反映当前仿真模型对两种代码形式的延迟记账差异。

---

### 2.3 维度 C：CModel 仿真框架微架构建模粒度审查 (Simulation Granularity)

我们深入剖析了 Gem5 仿真配置文件 `driver/pciemu/log/gem5/device1/config.ini`、`config.json` 以及执行日志：

1. **MPU 流水线仿真在配置中被显式禁用 (`pipeline=false`)**：
   在 `config.ini` 的 `[system.apc.ape0.mpu]` 节中明确声明：
   ```ini
   [system.apc.ape0.mpu]
   type=HPP2MPU
   pipeline=false
   mpu_hardware_debug=false
   check_mpu_wb_addr=false
   icache=true
   icache_size=262144
   ```
   **这意味着 MPU 根本没有启用周期精确的动态流水线模型**！MPU 在 CModel 中仅作为一个函数式指令执行器运行，并以静态硬编码的延迟推进仿真时间。

2. **片外内存系统使用简化的 `SimpleMemory`，无 cycle-accurate DRAM 控制器**：
   在 `config.ini` 中搜索内存类型，系统全部采用 gem5 的基础假象内存模块：
   ```ini
   type=SimpleMemory
   latency=30000  # 固定延迟
   ```
   整个工程内**不存在任何 `DRAMCtrl`**、DDR4/5 或 LPDDR 控制器。这意味着：
   - 仿真器没有对物理内存的 Row Buffer 命中/冲突（Row Buffer Hit/Miss/Conflict）进行建模。
   - 没有对 Rank/Bank 交错、刷新（Refresh）、tRCD/tCAS/tRP 物理时序进行建模。
   - DDR 访问在 CModel 中被抽象为一个粗粒度的固定常数延迟。

3. **微架构统计计数器缺失 (`stats.txt` 为空)**：
   - 仿真运行目录下的 `stats.txt` 大小始终为 **0 字节**。仿真器没有输出任何关于流水线停顿原因（Pipeline Stalls）、发射槽利用率（Issue Slot Utilization）、分支预测失败率或 Cache Miss 分类的硬件级遥测数据。
   - Gem5 启动日志中输出明确告警：
     ```text
     warn: instruction SEQ_prefetch_r not implemented!
     warn: instruction SEQ_set_jump_mode not implemented!
     warn: instruction SEQ_clean_cache not implemented!
     ```
     部分高级预取与缓存清洗指令在当前仿真器版本中甚至尚未实现。

---

## 3. 总体结论与定性 (Final Assessment)

综合上述三项调查，针对“每条向量 FMA 耗费约 100 周期”的疑问，给出明确的技术定性：

1. **性能归因定性**：
   - 在该模型的记账方式下，手写 VLIW 约 1 周期/op，编译 C 为 74 周期/op，差距归因于工具链未做软件流水。
   - 编译器 `scale-cc` (LLVM 12) 尚不具备将高级 C++ 循环自动映射为流水化 VLIW 指令包的能力，产生了大量的 NOP 气泡与栈溢出访存。
   - 本测试不对物理硅片的能力下任何判断。由于 CModel 关闭了流水线建模（`pipeline=false`）且使用固定延迟的 `SimpleMemory`，仿真器记账无法为真实硅片的吞吐能力作证。

2. **CModel 仿真器定性**：
   - **当前目标 LPU CModel 不属于周期精确（Cycle-Accurate）仿真器**。
   - 其本质是一个**指令集级/功能级仿真器附带粗粒度静态延迟估算模型**（ISA Functional Emulator with Coarse Latency Model），其中 MPU 流水线未建模（`pipeline=false`），DDR 仅为固定时延模型（`SimpleMemory`）。

3. **研究使用规范约束**：
   - **严禁再在任何报告、文档中将当前 CModel 称为“周期精确”（Cycle-Accurate）仿真环境**。
   - **严禁将当前 CModel 测得的绝对 cycle 周期数或算子绝对耗时当作目标 LPU 物理芯片的实际硬件极限能力对外提供**。
   - 报告中的所有评测数据必须标注为“**基于当前软件栈（scale-cc 编译器生成代码 + CModel 静态估算模型）的实测基准**”，所反映的是当前软硬件协同生态在高级语言下的早期编译效率，而非量产硅片的物理上限。
