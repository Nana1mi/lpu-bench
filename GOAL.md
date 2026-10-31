# GOAL：在 Aurora CModel 上测量 LLM 推理典型算子，论证该架构适合 LPU / AF 分离中的角色

> 本文件是给自主开发 agent 的长期目标说明。请完整阅读后再开始；每次恢复工作时先读 PROGRESS.md 和 BLOCKERS.md。

---

## 0. 一句话目标

在 Aurora 芯片的 CModel 仿真环境（QEMU + gem5）上，用 SCALE 编程模型实现并测量三类 LLM 推理典型算子（FFN/GEMM、decode attention、流水重叠消融），得到**可信的仿真时间数据**，最终产出一份基于数据的分析报告，回答：

> **Aurora 的 SPU/MPU 解耦 + 片上 DM 架构，更适合承担大模型推理中的哪一部分（FFN 侧 / attention 侧 / decode 阶段），收益有多大，瓶颈在哪里？**

这是一个本科毕设项目，背景是：业界推理系统正在走 PD 分离（Prefill/Decode）和 AF 分离（Attention/FFN，例如 NVIDIA 将 decode 中无状态的 FFN/MoE 放到 SRAM 型 LPU，有状态的 attention 留在 HBM 型 GPU）。本项目要用实测数据说明 Aurora 在这套分工中的定位。

---

## 1. 背景知识（必须理解）

### 1.1 芯片与软件栈
- 厂商：思朗科技（smartlogictech）。核心向量核 MPU 源自其 MaPU 架构。
- 软件栈：toolchain（Clang/LLVM，编译 .sc 文件）、maps-kmd（内核驱动）、maps-umd（用户态驱动）、maps-fw 等。Host 端 API 类似 CUDA，前缀为 maps（mapsMalloc / mapsMemcpyAsync / <<<>>> / mapsLaunchKernelExc）。
- 参考文档放在仓库 docs/ 下（由人类提供）：
  - SCALE_Programming_Guide —— 编程最重要的参考，写 kernel 前必读第 3、4、5 章
  - maps-umd包使用说明 —— 测试用例、环境变量
  - Installation_Document_2.0.x —— mapkg 安装命令
  - 搭建cmodel环境 —— 启动/销毁 QEMU 节点
  - SCALE_CModel_Installation_Summary —— 本机已部署环境的说明与踩坑记录

### 1.2 架构要点（来自 Programming Guide）
- APE = SPU + MPU：SPU 是乱序标量核，负责 DMA 搬运、同步、调度；MPU 是 VLIW 向量核，只做计算，只能访问 DM，只能由 SPU 调用。
- SPU 调 MPU 三种方式：async_call（默认）、sync_call、async_enqueue（异步任务队列，用于减少空泡；不可与前两者混用）。
- 4 个 APE 组成一个 APC（共享 DM），4 个 APC 组成一个 APC cluster。
- SIMD 而非 SIMT：warpSize == 1，单线程直接操作 v16f32 等 512bit 向量类型。
- 一个 block 最多 128 线程，必须落在一个 APC 内。
- 两种执行模式：N 模式（同 CUDA，线程可切换）；C 模式（block 仅 2 或 4 线程，保证并行、独占 APC，需用 mapsLaunchKernelExc 启动）。
- 内存：DDR 为 global memory（64 位地址）；DM 分 24 个 bank，用 __mem0~4__（local）、__shared__、__fix0~5__（指定 bank）、__acc__、__residency__ 显式管理。
- 数据搬运：memcpy_async（DMA）配合 barrier 等待，barrier_type 有 sync_read / sync_write / sync_read_write。

### 1.3 必须遵守的编程约束（违反会导致编译失败或运行时诡异错误）
- 指向 DDR 的 kernel 参数要加 __DDR 修饰；__device__ 数组当指针用时需 (__ptr64 T*) 强转。
- kernel 参数不能用 long / unsigned long，64 位整数一律用 long long / unsigned long long（host/device 对 long 位宽理解不同，会导致 launch 失败）。
- host 侧 mapsMalloc 得到的虚地址需要 mapsMemobjGetPhysicalAddress 转换后再传给 device（参见 demo 工程）。
- Local memory region 不能跳号（用三个就只能是 __mem0__ __mem1__ __mem2__）。
- 强烈建议（N 模式）：单个 memory region <= 32KB，local region 不超过 4 个，block 线程数 <= 32。
- 链接时加 --check-scale-mem 做内存限制检查。
- 用户写 MPU 汇编时不能破坏 BIUT15 / KI15，需以 mfetch:jump KI15 结束（尽量用 MPU C 写法，避免汇编）。
- N 模式下单个 kernel 使用的 barrier 不超过 8 个。

---

## 2. 环境

| 项 | 值 |
|---|---|
| 宿主机 | Windows 11 WSL2，Ubuntu 24.04 |
| 宿主机工作目录 | /home/rog/lpu-workspace/scale/ |
| driver 目录 | /home/rog/lpu-workspace/scale/driver |
| 客体机 (QEMU) | Ubuntu 22.04，kernel 5.15 |
| 共享目录 | 宿主机 /home/rog <-> 客体机 /home/emu（9p 挂载，双向实时可见） |
| 登录客体机 | cd driver && ./pciemu/release/ssh_emu.sh，或 ssh -p <端口> root@localhost |
| 工具链 | 客体机 /opt/maps/toolchain |
| 模拟器配置 | 默认 4 个 APC |

---

## 3. 硬性规则

1. **禁止编造数据。** 报告和图表里的每一个数字都必须来自 `results/` 下由脚本生成的原始 CSV，并能通过 `scripts/` 复现。没测出来就写"未测得"及原因。
2. **只认仿真时间，不认墙钟。** gem5 仿真极慢（8 个内存测试约 62 秒），宿主机墙钟时间不能作为性能数据，最多作为"实验耗时"记录。
3. **不得改动已安装的驱动/工具链/固件。** 禁止执行 `mapkg uninstall`、`mapkg install`、`dpkg -r/-P`、删除 `/opt/maps` 或 `/lib/firmware/sl`。环境坏了就写入 `BLOCKERS.md` 并停下等人类处理。
4. **正确性先于性能。** 每个 kernel 的每个版本都要与 host 侧参考实现（纯 C++）比对结果，误差在容差内才允许进入计时。
5. **规模要小。** 先在最小规模上跑通，再逐步扩大。单次运行超过 30 分钟的配置要先评估是否必要。所有远程命令加超时。
6. **保密。** `docs/` 是厂商内部文档，不得上传到任何公开位置，不得粘贴到公开 issue 或网页。
7. 遇到同一问题连续尝试 5 次仍失败，就把现象、尝试过的方法、错误日志写入 `BLOCKERS.md`，转去做其他不依赖它的任务，不要无限重试。
8. 频繁用 git 提交（每完成一个小步骤提交一次），commit message 写清楚做了什么。

---

## 4. 分阶段任务

### 阶段 0：跑通官方 demo（入口）
- 在工具链或驱动包里找到 Programming Guide 5.2.3 节的 SCALE 端到端 demo 工程（包含 `vaddu32` / `vmulu32` / `vsubu32` 这几个 kernel），复制一份到工作区，编译并运行，看到 4 个 `Test PASSED`。
- 读懂它的 makefile、host 代码、SPU 代码、MPU 代码，写一份 `notes/demo_walkthrough.md`，说明编译流程（`.sc` → host/spu/mpu 目标文件 → 链接），以及 host 分配内存、拷贝数据、启动 kernel、取回结果的完整调用链。
- 搞清楚 MPU 支持哪些向量数据类型（fp32 / fp16 / bf16 / int8 等）和矩阵/向量加速 intrinsic，方法是搜索工具链头文件（如 `aurora.h`、`scale.h`、`scale/` 目录）。写入 `notes/mpu_capabilities.md`。

**完成标准**：demo 能复现运行；两份笔记完成。

### 阶段 1：建立可信的计时方法（最关键，不能跳过）
调研并验证所有可能的计时来源，结论写入 `METHODOLOGY.md`：
1. **Host 侧 event API**：在头文件中搜索 `mapsEvent*`（类似 `cudaEventRecord` / `cudaEventElapsedTime`）；参考 `umd_test/event/` 下的用例。
2. **Device 侧周期计数器**：在头文件中搜索 clock、cycle、counter、timer 相关的 intrinsic 或内置变量。
3. **gem5 统计**：`mapkg create -l` 打开模拟器 log，查找 gem5 的 stats 输出（例如 `stats.txt`、`numCycles`、`simTicks`）。
4. **masight**：已安装的性能监控工具，调研其用法和输出。
5. 辅助环境变量：`MAPS_KERNEL_REPEATITION_COUNT=N` 可让一次 launch 重复执行 N 次；**测时一律设置 `MAPS_DUMP_GPU_DEBUG_INFO_ENABLE=false`**（该调试打印默认开启，会严重拖慢性能），并关闭其他 dump 选项，`MALOG_LEVEL` 设为 error。

**验证方法**（每个候选计时源都要做）：
- 对一个简单 kernel，在工作量为 1×、2×、4×、8× 时分别计时，检查结果是否近似线性；
- 同时记录宿主机墙钟时间，确认候选计时源反映的是**仿真时间**而不是宿主机时间。如果两者高度相关、比例恒定，就要怀疑它测的是宿主机时间；
- 同一配置重复运行 3 次，记录方差。

**产出**：`METHODOLOGY.md` 写明最终采用的计时源、计时区间（只包含 kernel 执行，还是包含拷贝）、单位换算（cycles 与时间，频率未知时直接用 cycles）、验证证据；以及一个可复用的计时封装（`common/timer.h` 或类似文件）。

**如果找不到任何能反映仿真时间的计时源**：立即写入 `BLOCKERS.md`，标明"需要人类向老师或厂商询问计时方法"，暂停阶段 2 的测量部分（但可以先实现 kernel 并做正确性验证）。

### 阶段 2：实现并测量三类算子
所有 kernel 放在 `kernels/<name>/` 下，每个版本都有 host 参考实现和正确性检查。数据类型优先选 MPU 原生支持的类型（先用 fp32 跑通；如果支持 fp16 或 int8，再加一组）。

#### K1：FFN / GEMM（计算密集，对应 AF 分离中的 FFN 侧）
- 计算 `Y = X · W`，其中 X 为 `[M, K]`，W 为 `[K, N]`。M 代表 batch 内的 token 数。
- 先实现分块 GEMM：权重分块经 DMA 搬入 DM，MPU 做分块乘加，结果写回 DDR。
- 版本：
  - **v0 serial**：搬一块、算一块、等待，严格串行；
  - **v1 overlap**：DMA 双缓冲（当前块计算时预取下一块）+ `async_enqueue` 连续投递 MPU 任务。
- 扫描：M ∈ {1, 2, 4, 8, 16, 32, 64}（受仿真速度限制可以裁剪）；K、N 选 2~3 组小尺寸（例如 256、512），以实际能跑完为准。
- 指标：cycles、FLOPs/cycle、利用率（相对理论峰值；如果查不到峰值，就以实测最佳值为参照，并注明）。
- 要回答的问题：利用率是否随 M 增大而上升？在多大 M 处饱和？

#### K2：decode attention（访存密集，对应 AF 分离中的 attention 侧）
- 单个 query 向量（decode 一步），对长度为 L 的 KV cache 计算 `softmax(q·Kᵀ/√d)·V`，head_dim d ∈ {64, 128}。
- 版本：
  - **v0 KV in DDR**：KV 留在 DDR，每次通过 DMA 流式搬入 DM；
  - **v1 KV in DM**：在 DM 容量允许的范围内，KV 预先常驻 DM（模拟 SRAM 驻留），只计时计算部分；
  - **v2 overlap**：KV in DDR，但用双缓冲重叠 DMA 与计算。
- 扫描：L ∈ {64, 128, 256, 512, 1024, …}，扩到 v1 放不下为止，并记录 DM 能容纳的最大 L。
- 指标：cycles、有效带宽（字节/cycle）、计算时间与搬运时间的占比。
- 要回答的问题：KV 不在片上时，搬运占多大比例？重叠能挽回多少？

#### K3：流水重叠与 bank 布局消融
- 汇总 K1、K2 的 v0 与 overlap 版本，计算加速比，量化 SPU/MPU 解耦带来的收益。
- 可选：对 K1 做 bank 布局消融，对比 `__fix__` 显式分配、编译器默认分配、刻意让两个热点数组落在同一 bank 这三种情况。
- 可选：对比 N 模式与 C 模式（C4）。

**每个 kernel 的完成标准**：所有版本在所有配置下通过正确性检查；计时数据写入 CSV；扫描完成或明确记录了裁剪原因。

### 阶段 3：实验框架
- `scripts/run_sweep.py`：读取配置（YAML 或 JSON），在宿主机上通过 ssh 驱动客体机编译和运行，每个配置重复 3 次，把结果追加写入 `results/<kernel>.csv`。字段至少包括：kernel、version、各尺寸参数、dtype、repeat 序号、cycles、正确性、git commit hash、时间戳。
- 支持断点续跑：已有结果的配置跳过。
- `scripts/plot.py`：在宿主机上用 matplotlib 从 CSV 生成图表，输出到 `figures/`。至少包括：
  1. K1 利用率随 M 变化的曲线（v0 与 v1）；
  2. K2 cycles 随 L 变化的曲线（三个版本）；
  3. K2 计算/搬运时间占比堆叠图；
  4. K3 各算子的 overlap 加速比柱状图。
- 如果能得到或估算峰值算力与 DDR 带宽，再画一张 roofline 图，标出各算子的位置；拿不到就不画，并在报告中说明。

### 阶段 4：报告
写 `REPORT.md`（中文），结构如下：
1. 背景：PD 分离、AF 分离、LPU 概念，以及本研究的问题；
2. Aurora 架构简介（SPU/MPU、DM、执行模式），只用自己的话概括，不要大段照抄 docs；
3. 实验方法：计时方法及其验证、实验平台、算子实现要点；
4. 结果：上述图表，每张图配 2~3 句数据解读；
5. 分析与结论：直接回答第 0 节的核心问题；
6. 局限性：仿真与真实芯片的差异、规模限制、未能测量的部分；
7. 复现说明：一条命令或一个脚本跑出全部结果。

---

## 5. 仓库结构

```
lpu-bench/                  # 放在宿主机共享目录下，例如 /home/rog/lpu-workspace/lpu-bench
├── GOAL.md                 # 本文件
├── PROGRESS.md             # 工作日志：日期、做了什么、结论、下一步
├── BLOCKERS.md             # 需要人类介入的问题
├── METHODOLOGY.md          # 计时方法与验证
├── REPORT.md               # 最终报告
├── docs/                   # 厂商文档（不提交到公开远程仓库）
├── notes/                  # 调研笔记
├── common/                 # 公共代码：计时封装、正确性检查、host 参考实现
├── kernels/
│   ├── gemm/
│   └── decode_attn/
├── configs/                # 扫描配置
├── scripts/                # guest_run.sh、run_sweep.py、plot.py
├── results/                # 原始 CSV 与运行日志
└── figures/
```

`.gitignore` 中排除 `docs/`；如果要推送到远程，只能推送到私有仓库。

---

## 6. 工作协议

- **每次开始**：读 `GOAL.md`、`PROGRESS.md`、`BLOCKERS.md`，然后做环境健康检查。
- **每完成一步**：更新 `PROGRESS.md`（日期、完成内容、关键发现、下一步），并 git commit。
- **不确定 API 用法时**：先查 `docs/`，再搜索工具链头文件和 demo 源码，最后才是试错。把查到的结论记入 `notes/`，避免重复查找。
- **遇到以下情况，停下来写入 `BLOCKERS.md` 等待人类**：
  - 环境损坏，需要重装驱动或工具链；
  - 找不到任何反映仿真时间的计时源；
  - 文档与实际行为严重不一致，并且影响实验结论；
  - 需要芯片参数（峰值算力、DDR 带宽、主频），而头文件和文档里都找不到。

---

## 7. 完成标准（Definition of Done）

全部满足才算完成：
- [x] 阶段 0：demo 能复现运行；`notes/demo_walkthrough.md` 和 `notes/mpu_capabilities.md` 完成
- [ ] 阶段 1：`METHODOLOGY.md` 写明经过验证的仿真时间计时方法
- [ ] K1 GEMM：v0、v1 两个版本正确性全部通过，完成 M 扫描
- [ ] K2 decode attention：v0、v1、v2 三个版本正确性全部通过，完成 L 扫描
- [ ] K3：加速比汇总完成
- [ ] `results/` 中有完整的原始 CSV；`figures/` 中有至少 4 张图，全部由脚本生成
- [ ] `REPORT.md` 完成，第 5 节明确回答核心问题，所有数字都能追溯到 CSV
- [ ] 存在一键复现脚本，并在干净的检出上实际跑过至少一个小配置
- [ ] `BLOCKERS.md` 中没有未处理的阻塞项，或者每一项都已在报告"局限性"一节中说明
