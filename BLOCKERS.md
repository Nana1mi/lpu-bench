# 阻塞与待解决问题清单 (BLOCKERS.md)

本文档记录本项目在目标 LPU 处理器 CModel 仿真评测过程中遇到的阻塞项、未知参数及工具链/驱动层异常。根据 `GOAL.md` 规范，所有无法独立在实验室内部解决的技术阻碍均在此归档，供后续与芯片厂商团队及人类研究员沟通确认。

---

## 1. 芯片关键物理参数确认与体系结构全貌跟进（部分解决）

- **进展状态**：
  1. **厂商书面答复记录**：标称主频 1.425 GHz，每周期发射 8 条 `v16fp32` FMA，8 路 GDDR6 总带宽 512 GB/s，8 卡双 Full Mesh 互联带宽 1024 GB/s；
  2. **厂商官方算子库代码硬编码参数 (`device_info.h`, `matmul_perf_info.h`)**：
     - 主频：1.6 GHz；
     - 物理整卡规格：2 Dies × 32 APCs/die × 4 APEs/APC = 64 APCs (256 APEs)；
     - 理论物理峰值带宽：3276.8 GB/s；
     - 理论计算峰值：FP32 为 52.5 TFLOPS，FP16 为 420 TFLOPS，INT8/FP8 为 840 TFLOPS。
- **当前转入跟进的细化事项**：
  1. **低精度计算单元指令调用规范**：FP16 与 INT8 张量算力的指令助记符与高级语言内联接口；
  2. **片间互联通信延迟**：8 卡双 Full Mesh 下的物理传输与路由跳步延迟；
  3. **CModel 仿真校准参数**：CModel 关闭 MPU 流水线（`pipeline=false`）下的物理时延对齐基准；
  4. **官方微码算子库真实吞吐基准**：微码与汇编相比高级 C 编译的代码性能基准。
- **需要人类做什么**：
  在后续技术对接中向芯片厂商团队获取上述细化规格。

---

## 2. 片上高速数据存储 (DM) 真实物理容量与分区上限机制（部分解决）

- **现象**：
  先前由于将多 Warp 编程规范中的“建议 $\le 32\text{ KB}$”误判为硬件单 Bank 上限，导致 Attention 常驻评测仅做到 $L=128$。
  经实测与文档核实，已确定：
  1. 单个 APC 包含 **24 个独立 Bank**，单 Bank 物理容量为 **256 KB**，单 APC 物理 DM 总量为 **6 MB**。
  2. 在单 APE 下，单 Region 物理上限为 **256 KB**，实测支持 $L=1024$ 完整常驻（耗时 2,152,767 cycles）。
  3. 当申请超过 256 KB（例如 $L=2048$ 对应 512 KB）时，离线编译链接能通过，但驱动运行时 JIT 链接时精确抛出：`ld.lld: error: section .MEMORY0 size 524288 bytes exceeds max DM size 256KB`（$L=1536$ 对应 384 KB，报 393216 bytes），导致启动失败。
- **仍需人类做什么**：
  单 APC 虽有 6 MB DM，但当前编译器与链接器将每个 `__mem__` 严格绑定到单一 Bank（上限 256 KB）。需向厂商系统软件团队确认：是否存在通过自定义链接脚本、`__shared__` 或平坦寻址修饰符将多个 Bank 聚合申请超过 256 KB 连续缓冲区的编程机制。

---

## 3. `mapsEventElapsedTimeEx` 驱动计时 API 返回值语义不明

- **现象**：
  驱动层提供的硬件事件计时 API `mapsEventElapsedTimeEx(&ms, start, end)` 返回的毫秒数值与事件计数值逐行恒等（例如 `ms = 3950.0000`, `cycles = 3950`）。且 `wall_ms - sim_ms` 在不同负载下均恒定在 50~90 ms 之间，表现为宿主机墙钟耗时减去固定的 host 侧 memcpy/同步开销。该指标反映的是宿主机运行时间而非芯片仿真内部时间，无法作为独立的硬件性能度量，已从性能度量口径中剔除。
- **已尝试的查找方法**：
  排查了 `driver/pciemu/log/` 下的驱动日志，观察到 `imapsEventElapsedTimeEx: ... Elapsed Time = , 856, Elapsed Cycle = , 856`。确认驱动底层将同一时间量直接赋给了时间与周期字段。
- **需要人类做什么**：
  向芯片厂商软件驱动团队反馈该现象，确认 `mapsEventElapsedTimeEx` 在 CModel 仿真模式下是否仅作为 mock 桩实现。

---

## 4. `scale-cc` 编译器代码生成效率缺陷与 CModel 流水线模型未开启

- **现象**：
  `scale-cc` 编译器在将高级语言 C 编译为 MPU 汇编时，存在严重的静态代码生成低效：
  1. 无法自动融合乘加操作，强制将 `a * b + c` 拆解为 `vmul + vadd`，算力直接折半；
  2. 缺乏软流水（Software Pipelining）与循环展开优化，向量计算与片上 SDMA 数据搬运串行执行，导致计算与访存无法重叠；
  3. 编译器插入大量冗余的栈同步指令（`s_st_t15 / s_ld_t15`）与 NOP 气泡，实测单向量 FMA 耗时 74.00 周期/op，比手写汇编（1.003 周期/op）慢约 74 倍。
  同时，当前 CModel 仿真环境中的 gem5 配置显式关闭了 MPU 流水线仿真（`pipeline=false`），导致所测得的 cycle 数主要受制于编译器生成质量，无法直接代表物理硅片的执行上限。
- **已尝试的查找方法**：
  对 `kernels/gemm` 与微基准 `kernels/test_fma` 进行了 MPU 汇编解包与反汇编分析（`notes/cmodel_mpu_granularity.md`），建立了单向量指令周期线性回归模型（$R^2 = 1.00000$）。
- **需要人类做什么**：
  向芯片厂商编译器团队咨询是否具备开启向量循环展开与软件流水调度的优化开关，或获取官方高性能数学库（BLAS / Attention）的手写汇编实现；向仿真器开发团队咨询未来版本是否计划支持周期精确的 MPU 动态流水线与 DRAM 冲突仿真。

---

## 5. Attention 算子扩展至 head_dim d=128 时遭遇 MPU 寄存器池耗尽与栈溢出仿真停滞（T4 调查结论）

- **现象**：
  根据 `GOAL.md` K2 任务规范，注意力算子应评估 $d \in \{64, 128\}$。然而在目标 LPU 的 MPU 架构中，每个 Warp 仅配备 16 个 512 位宽向量寄存器（$V_0 \sim V_{15}$）。
  在 $d=64$ 时，每个 Token 的向量宽度为 4 个 512-bit 向量，$Q$ 占用 4 个寄存器，尚有 12 个寄存器用于承载 $K$ 分块及中间内积累加；
  当扩展至 $d=128$ 时，$Q$ 独占 8 个向量寄存器，若对 $K, V$ 进行两路分块展开或流水计算，16 个向量寄存器立即耗尽。`scale-cc` 编译器在处理多向量长依赖时会强制触发向栈存储区（`T15[15]`）的频繁保存与恢复（register spill & reload）。
  在 gem5 指令级全系统仿真环境下，大量的栈访存引发了极端严重的解释执行放大效应，单步模拟时延呈数十倍暴增，单次测试在宿主机需持续消耗数小时以上，在全自动化基准测试套件中极易导致超时中断。
- **已尝试的方法与最新调查进展**：
  1. 测试将 `decode_attn.sc` 的 `D_MODEL` 改为 128 进行编译运行，证实由于寄存器溢出导致指令数暴增，仿真出现停滞；
  2. 针对 $d=128$ 尝试按 64 维分两趟分块计算累加，但高级 C 编写在经过 LLVM 编译后，编译器依然无法识别跨步生命周期，中间标量指针与索引依然引发高频栈溢出；
  3. 核查厂商官方 Demo（`demo4scalee2e/maops/third_party/flashattention/`）源码证实：官方 FlashAttention 在处理大维度时**完全采用手写汇编（`fa_*.m.asm`）精细控制 16 个向量寄存器复用与双缓冲**，彻底绕过了高级 C 编译器。根据 `GOAL.md` 规则 7，当前 C 层面尝试已达上限，转入阻塞状态，不再无谓消耗仿真机时。
- **需要人类做什么**：
  向芯片厂商团队获取官方手写的 $d=128$ 向量优化微码模板或 FlashAttention 汇编内核。

---

## 6. Git 历史记录涉密物料与厂商标识清洗（已解决：已完成全库脱敏与历史重写）

- **现象与处理结果**：
  1. 早期提交历史中曾包含厂商专有示例文件（`kernels/official_demo/`、`maops.h`、`*.m.asm`、驱动日志 `*malog*.txt`、`*scalehost_log*.txt` 与 `results/maps_event_driver_log_excerpt.txt`）以及厂商与芯片代号字样；
  2. 经人类研究员授权确认，已完成全库文档、图表、脚本及源码的统一学术化脱敏（源码通过 `common/lpu_arch.h` 预处理器宏与 `Makefile` 动态路径探测保持 100% 原生可编译性），并通过孤儿分支（orphan branch）重写/压缩 Git 历史并强制更新至远端，彻底消除了历史 commit 快照与 commit message 中的涉密物料与厂商标识。
- **CSV 历史 `commit_hash` 溯源说明**：
  `results/*.csv` 中记录的 `commit_hash` 字段为各阶段物理实验采样时的原始快照哈希，严格遵循 `METHODOLOGY.md`“严禁回填或篡改已采样 CSV 历史行”的规范保留原值，可结合 `build_id`（ELF 二进制 MD5）进行机器核验。同时建议远端 GitHub 仓库继续保持 Private 状态。

---

## 7. 服务器 17 交叉验证依赖外部协作推进（T3）

- **现象**：
  为确认 WSL 仿真环境与服务器 17 实体仿真环境的数据一致性，需将编译好的二进制送交服务器 17 运行比对 `device_cycles`。
- **准备工作**：
  已在 `notes/cross_validation_server17.md` 中固定了待比对的二进制文件及 MD5（`gemm_multicore.out`: `bbf153e9b289b92bcf8055794e812953`，`bank_ablation.out`: `8792b2ed03a7acc36c99c2eca52f10dd`，`test_timer.out`: `d81ca9cb110eb7efa1affbfc3e3453a4`），并列出对齐基准（`test_timer`: 1,000 cycles；`gemm_multicore M=1 N1`: 648,903 cycles）。
- **需要人类做什么**：
  联系服务器 17 负责同学，在服务器 17 上拉取上述二进制执行对应测试命令，并反馈 `device_cycles`；若结果存在分歧，按 `notes/cross_validation_server17.md` 排查两边的 `driver/pciemu/log/gem5/device1/config.ini`。

---

## 8. 多核跨 APE Memory Cluster 硬件争用微基准设计（待推进）

- **背景与澄清**：
  先前在单核下将两个输入数组修饰为同一 `__fix1__` 属于逻辑地址空间重叠（别名 Bug），数据互相覆盖导致内核未启动即异常终止（`device_cycles: 0`），并非硬件冲突评测。
  厂商头文件 `device_info.h` 明确给出了硬件真实的冲突机制：单 APC 内部 `DM(i)` 与 `DM(i+12)` 属于同一 Memory Cluster；APE0 与 APE1（或 APE2 与 APE3）并发访问同一 Cluster 会触发硬件流水线 Stall，且同一 Cluster 内共享累加单元，并发写回 `Store(WB)` 亦会发生争用。
- **阻塞原因与实现要求**：
  该硬件争用属于**多核跨 APE 的物理资源冲突**，必须设计精细协调的多线程内核（如 APE0 绑定 DM0、APE1 绑定 DM12 并发密集读写），在 C 层面精确构造无数据依赖的并发争用模式以测量 Stall 周期开销。由于当前精力集中在单 APC 并行扩展基线，该跨核硬件冲突微基准作为待推进事项归档在此。
- **需要人类做什么**：
  确认后续是否需要开发跨 APE 针对性冲突内核以精确量化 Memory Cluster 仲裁时延。

---

## 9. 硬件标称参数存在两套来源的分歧与矛盾（待厂商澄清）

- **现象与三处关键矛盾**：
  在厂商初期书面答复（2026-11-03）与官方算子库 Demo 源码硬编码常量（`device_info.h` / `matmul_perf_info.h`）之间，存在三处显著的硬件参数分歧：
  1. **频率矛盾**：初期书面答复标称为 **1.425 GHz**；官方代码库硬编码为 `k_spu_main_frequency = 1.6 * 1e9`（**1.6 GHz**），差异 +12.3%；
  2. **APC / APE 规模矛盾**：初期答复未标明物理整卡规模，CModel 仿真器默认配置为 **4 APC (16 APE)**；官方代码库硬编码为 `k_die_per_card = 2`、`k_apc_num_per_die = 32`、`k_ape_num_per_apc = 4`，物理整卡为 **64 APC (256 APE)**，单 Die 即达 32 APC (128 APE)，是 CModel 默认规模的 16 倍；
  3. **片外内存物理带宽矛盾**：初期书面答复为 **8 路 GDDR6 @ 512 GB/s**；官方代码库硬编码为 `k_max_memory_bandWidth_gbs = 3.2 * 1024`（**3276.8 GB/s**），高出 6.4 倍，已达 HBM 级带宽量级。
- **影响与应对原则**：
  **参数存在两套来源，结论可能随澄清而变**。目前项目微基准评测核心指标均以硬件机器周期（`device_cycles`）与每周期浮点能力（FLOPs/cycle）为准，具备主频与规模不变性；涉及物理时间折算与 Roofline 带宽天花板时，项目在报告与备忘录中并列呈现两套参数体系，明确标注出处与推导逻辑。
- **需要人类做什么**：
  向芯片厂商团队索取官方正式芯片硬件规格书（Datasheet），澄清：
  1. 芯片量产标准主频与升频范围；
  2. 实体加速卡内存形态（GDDR6 还是 HBM/多堆叠）及其实际物理峰值带宽；
  3. 物理整卡 Die/APC 拓扑结构。

---

## 10. 厂商算子库测试套件 `libmablasLt_perf` 宏旁路 DMA 与插桩边界异常（待排查）

- **现象**：
  在对 `mablaslt` 矩阵乘内核（`gemm_matmul_cmode` / `gemm_slice_k_matmul_cmode`）插入 `tick()` 测量硬件周期时，观察到 $M \in [1, 64]$ 内 FP32 周期恒定在 ~108,650 周期、FP16 恒定在 ~51,500 周期，且存在 $\pm 80\sim 110$ 周期抖动（与自研内核在同一 CModel 下方差为 0 的表现不符）。
  定性诊断实验（已落盘于 `results/vendor_gemm_diagnostic.csv`）表明：固定 $M=8$ 将 $K, N$ 从 256 扩展至 512（计算量增大 8 倍，A/B/C 数据量增大 4 倍），实测 `device_cycles` 依然为 107,563 / 107,665 恒定不动，未随计算规模扩展。
- **根因分析**：
  查阅厂商 CMake 与源码发现：
  1. `blas_perf` 测试程序链接的是 `libmablasLt_perf.so`，其编译时注入了 `-DBLAS_PERF_SHARED_LIB` 宏；
  2. 在 `llcd_csu_dma.h`（第 48 行）与 `llcd_dma.h`（第 306 行）中，所有实际的 DDR $\leftrightarrow$ DM 数据搬运（`deviceMemcpyPaddingAsync` / `deviceMemcpyAsync`）均置于 `#ifndef BLAS_PERF_SHARED_LIB` 条件编译块内。在 perf 动态库下，DMA 搬运被完全宏旁路（No-Op）；
  3. 内核调度传入的是固定编译期模板常量（`dev_tile.M_THD_TILE_A = 128`），测得的周期实际仅包含一段不带数据搬运与外层动态调度的固定计算循环；$\pm 100$ 周期抖动源于 `SYNC_MPU_KERNEL` barrier 轮询；
  4. **精度校验字段同样无效与推理更正**：`blas_perf.cpp` 本身不执行任何针对 CPU 参考结果的正确性比对（此前 CSV 中的 `correct=PASS` 与 `max_diff=0.000000e+00` 系采集脚本硬编码填入的占位符）；在 DMA 宏旁路下，真实数据的搬入、计算与写回根本没有执行，此前“以 `max_diff=0` 推断计算已完成但未计入”的推理不成立，周期恒定的根因统一归结为 **DMA 宏旁路与编译期固定瓦片空转**；
  5. 基于该固定周期推导出的吞吐率随 $M$ 线性增长属于除法算术表象，相关指标及 `correct` / `max_diff` 已全量标注 `[待重新测量]`，且 `results/vendor_gemm.csv` 与 `results/vendor_gemm_diagnostic.csv` 内部已清空 `correct` / `max_diff` 并新增 `validity=invalid_dma_bypassed` 自描述列。
- **需要人类做什么 / 应对策略**：
  需在后续测试中脱离 mock DMA 的 `libmablasLt_perf`，切换至调用真实动态库 `libmablasLt.so` 的程序（如 `sample/mablasLt/00_basic_matmul.cpp`），并在包含完整 DMA 搬运与外层 Tile 展开的 Host/Device 顶层边界重新进行硬件周期插桩与单调性准入验证（详见条目 11）。

---

## 11. 真实库 `00_basic_matmul` (`libmablasLt.so`) 在 T3 单调性门禁测试中的固定瓦片与多 APC 切分行为解释（已解决：`valid_with_explanation`）

- **现象**：
  按照任务书 T1–T3 要求，切换至链接真实库 `libmablasLt.so`（不含 `-DBLAS_PERF_SHARED_LIB`，`build_id = 41a1645b9e143f255310c8b65a0c798b`）且自带 CPU 参考精度校验的 `00_basic_matmul`，在 `gemm_matmul_cmode`（`general_matmul_tile.sc:83-120`）入口 `__syncthreads()` 后、`matmulkernel.init()` 与 `matmulkernel.process()` 之前记录 `t_start = tick()`，在 `matmulkernel.process()` 完成后记录 `t_end = tick()`，并将差值写回输出缓冲末尾预留区 `h_matD + B*M*N`。
  在 QEMU CModel 上实测 T3 单调性门禁与补充验证数据（已落盘于 `results/vendor_gemm_real_gate.csv`）：
  1. **基线组 $(M=8, K=256, N=256)$**：3 次重复 `device_cycles` 为 `76,076 / 75,969 / 75,996`（均值 **`76,013.67`**，极差 107，`correct=PASS`，`max_diff=0.000000e+00`）。注：两个构建（`libmablasLt.so` 与 `libmablasLt_perf.so`）的编译宏与插桩边界均不同，两组周期不具可比性，不作对比；
  2. **$M$ 翻倍交叉验证组 $(M=16, K=256, N=256)$**：3 次重复为 `76,241 / 76,239 / 76,240`（均值 **`76,240.00`**，极差 2，`correct=PASS`，`max_diff=0.000000e+00`），较 $M=8$ 严格单调增加 **+226.33 周期**（真实 DMA 搬运量翻倍）；
  3. **$K, N$ 翻倍门禁组 $(M=8, K=512, N=512)$**：3 次重复为 `77,909 / 77,991 / 77,911`（均值 **`77,937.00`**，极差 82，`correct=PASS`，`max_diff=0.000000e+00`），较 $(8, 256, 256)$ 增加 **+1,923.33 周期**，比值为 **`1.0253×`**；
  4. **$M=512$ 与 $N=1024$ 补充验证组**：
     - $(M=512, K=256, N=256)$：3 次重复为 `81,705 / 81,744 / 81,709`（均值 **`81,719.33`**，极差 39，`correct=PASS`，`max_diff=0.000000e+00`），较 $M=8$ 增加 **`+5,705.67` 周期（`1.0751×`）**；
     - $(M=8, K=256, N=1024)$：3 次重复为 `75,996 / 76,076 / 75,970`（均值 **`76,014.00`**，极差 106，`correct=PASS`，`max_diff=0.000000e+00`）。

- **根因定位与门禁通过裁定（源码级三层机制 + 峰值合理性旁证）**：
  1. **启发式预测桩函数硬编码导致无条件选用 `B1M256N256K512` (`general_matmul_register.cpp:215-226`)**：
     `mablasLt` 注册了 `B1M256N256K256`（$K_{tile}=256$）和 `B1M256N256K512`（$K_{tile}=512$）两个 FP32 算子，但 `MABLASLT_IMPL_KERNEL_PREDICT_PERF` 为未完成的桩函数（`// TODO(Jiahao): add perfInfo`），分别硬编码返回 `1e-8` 与 `1e-9`。因此启发式选择器无条件选用单瓦片为 $256 \times 256 \times 512$ 的 `B1M256N256K512`；
  2. **汇编微码核编译期固定 $K_{tile}=512$ (`general_matmul_kernel.h:75,83,346-352`)**：
     在 `B1M256N256K512` 内核中，每个 APE 调度的微码核 `matmul_mpu_kernel_.launch_sync` 接收编译期 `constexpr` 常量 `M_THD_TILE_A=64, N_THD_TILE_B=64, K_THD_TILE_A=512`。当 $K=256$ 时，`LlcdCsuDmaReadMatA/B` 对 $K \in [256, 512)$ 硬件补零，MPU 汇编微码核仍完整执行 512 步 K 循环（约 74,000 周期）。故 $K: 256 \to 512$ 时，`k_loops_ = CEIL_DIV(K, 512)` 恒为 1，MPU 计算周期增量为 0，仅增加真实 DMA 搬运耗时；
  3. **多 APC 调度器按 `k_apc_num_per_die = 32` 自动跨 APC 切分 $M$ 与 $N$ (`general_matmul_info.h:113-141`, `device_info.h:16`)**：
     `get_mm_tile_split_k` 在总瓦片数 $\lceil B/1 \rceil \times \lceil M/256 \rceil \times \lceil N/256 \rceil \times \lceil K/512 \rceil \le 32$ 时，直接将 $M$ 与 $N$ 方向瓦片分配给多个并行 APC（`m_blk_num = CEIL_DIV(M, 256)`，`n_blk_num = CEIL_DIV(N, 256)`）。因此在 $(8, 512, 512)$（`n_blk_num = 2`）、$(512, 256, 256)$（`m_blk_num = 2`）以及 $(8, 256, 1024)$（`n_blk_num = 4`）下，单 APC 内的 `m_loops_` 与 `n_loops_` 均保持为 1，单 APC 周期增量严格反映该 APC 负责的真实 DMA 载荷变化（如 $M=512$ 时单 APC 搬运满 256 行 $A$ 与 $D$，周期较 $M=8$ 增加 `+5,705.67`）；
  4. **物理峰值合理性旁证**：
     在 $(8, 256, 256)$ 下（1 个 APC = 4 个 APE），单瓦片 $256 \times 256 \times 512$ 的实际硬件浮点运算量为 $2 \times 256 \times 256 \times 512 = 67.11\text{ MFLOP}$，除以实测 `76,014` 周期得单 APC 硬件实际吞吐 **`882.9 FLOPs/cycle`**，相当于单 APC FP32 理论峰值（$4 \times 256 = 1024\text{ FLOPs/cycle}$）的 **86.2%**（若按 MPU 纯计算段约 74,000 周期计则达 **88.6%**）；在 $(512, 256, 256)$ 下（`m_blk_num = 2`，2 个并行 APC，双 APC 理论峰值 $2 \times 1024 = 2048\text{ FLOPs/cycle}$，实测 `81,719.33` 周期），双 APC 合计有用功吞吐（$K=256$）为 $821.2 \div 2048 = \mathbf{40.10\%}$，双 APC 合计硬件实际功吞吐（含 $K_{tile}=512$ 空转）为 $1642.4 \div 2048 = \mathbf{80.20\%}$（等价于单 APC 硬件实际功 $821.2 \div 1024 = 80.20\%$），证明计时点准确覆盖了真实硬件执行区间。

- **`00_basic_matmul` 入口使用边界说明（非库缺陷）**：
  在诊断测试 $(M=8, K=1024, N=512)$（`results/vendor_gemm_real_gate.csv` 第 10 行）时，由于 $K=1024 > 512$，`get_mm_tile_split_k` 触发了跨 APC Split-K 切分（`k_blk_num = 2, n_blk_num = 2`），此时各 APC 将部分和写入临时 workspace，需要随后调用二级 Split-K 规约 kernel 才能将结果写回输出张量 $D$。由于示例程序 `00_basic_matmul` 仅封装了单次主算子启动、未调用后续规约 kernel，导致输出校验失败（`correct=FAIL, max_diff=1.007412e+03`）。**因此，使用 `00_basic_matmul` 入口测试时应保持 $K \le 512$，以避免触发跨 APC Split-K；这属于示例入口的使用边界，而非 `libmablasLt.so` 库缺陷。**

- **解决结论**：
  T3 门禁判定为 **通过（`valid_with_explanation`）**，T4 采集采用原版未修改的 `libmablasLt.so`（方案 A，as-shipped 行为），相关门禁记录已标注 `validity=valid_with_explanation` 并附完整源码行号说明。



