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

1. 禁止编造数据。报告和图表里的每一个数字都必须来自 results/ 下由脚本生成的原始 CSV。
2. 只认仿真时间，不认墙钟。gem5 仿真极慢，宿主机墙钟时间不能作为性能数据。
3. 不得改动已安装的驱动/工具链/固件。
4. 正确性先于性能。每个 kernel 的每个版本都要与 host 侧参考实现（纯 C++）比对结果。
5. 规模要小。先在最小规模上跑通，再逐步扩大。所有远程命令加超时。
6. 保密。docs/ 是厂商内部文档，不得上传到任何公开位置。
7. 遇到同一问题连续尝试 5 次仍失败，写入 BLOCKERS.md，转去做其他任务。
8. 频繁用 git 提交。

---

## 7. 完成标准（Definition of Done）

- [ ] 阶段 0：demo 能复现运行；notes/demo_walkthrough.md 和 notes/mpu_capabilities.md 完成
- [ ] 阶段 1：METHODOLOGY.md 写明经过验证的仿真时间计时方法
- [ ] K1 GEMM：v0、v1 两个版本正确性全部通过，完成 M 扫描
- [ ] K2 decode attention：v0、v1、v2 三个版本正确性全部通过，完成 L 扫描
- [ ] K3：加速比汇总完成
- [ ] results/ 中有完整的原始 CSV；figures/ 中有至少 4 张图，全部由脚本生成
- [ ] REPORT.md 完成，第 5 节明确回答核心问题，所有数字都能追溯到 CSV
- [ ] 存在一键复现脚本，并在干净的检出上实际跑过至少一个小配置
- [ ] BLOCKERS.md 中没有未处理的阻塞项，或者每一项都已在报告一节中说明
