# 目标 LPU 芯片片上存储 (DM) Bank 映射规则与消融备忘录 (bank_mapping_rules.md)

本文档系统整理目标 LPU 芯片片上高速数据存储 (Data Memory, DM) 的底层硬件拓扑、物理 Bank 划分机制、编译器映射修饰符规则，以及基于实测数据的消融结论。

---

## §1 硬件底层拓扑与 Bank 物理划分

结合厂商算子库头文件（`device_info.h`、`matmul_perf_info.h`）与驱动底层链接脚本（`apc-fa-cmode.ld`、`dm_allocator.h`）核对：

1. **核与集群层级架构**：
   - **物理整卡规格**：单卡包含 **2 个 Die**，每个 Die 包含 **32 个 APC (Processing Cluster)**，每个 APC 包含 **4 个 APE (Processing Element)**，整卡物理规格共 **64 个 APC (256 个 APE)**。
   - **CModel 仿真实例**：当前本地 CModel 仿真环境默认实例化了 **4 个 APC (共 16 个 APE)**。

2. **单 APC 物理 Bank 组成**：
   - 每个 APC 配备 **24 个独立物理 SRAM Bank**（`k_max_dm_num = 24`），单个 Bank 的物理容量为 **256 KB**（`k_max_dm_byte_size = 256 * 1024`），单 APC 物理 DM 总容量为 **6 MB**（24 × 256 KB）。
   - 24 个 Bank 在 4 个 APE 间划分：每个 APE 支配 6 个物理 Bank（共 1.5 MB 物理空间）：
     - **常规存储空间 (2 Banks)**：`__fix0__` 对应各 APE 的首个 Bank（APE0: `DM0`, APE1: `DM6`, APE2: `DM12`, APE3: `DM18`，通常用作栈与标量/局部存储）；`__fix1__` 对应各 APE 的第二个 Bank（APE0: `DM1`, APE1: `DM7`, APE2: `DM13`, APE3: `DM19`）；
     - **累加存储空间 (4 Banks, ACC DM)**：`__fix2__` ~ `__fix5__` 对应支持 ACC 硬件累加的 Bank（APE0: `DM2~5`, APE1: `DM8~11`, APE2: `DM14~17`, APE3: `DM20~23`）。

---

## §2 编译器语法修饰符与映射规则

在 SCALE 编程语言中，片上 DM 的分配通过局部变量属性修饰符指定：

1. **编译器管理型修饰符 (`__mem0__` ~ `__mem4__`)**：
   - 语法：`__mem0__ v16fp32 buf[N];`
   - 语义：声明虚拟 Region 编号（Region 0 ~ 4）。链接器在编译期为每个虚拟 Region 分配空闲物理 Bank。
   - 约束：
     - 单个 kernel 内的 Region 编号必须连续（从 0 开始），不可跳号；
     - 单个虚拟 Region 的申请尺寸不得超过单 Bank 物理上限 256 KB（超过 256 KB 将触发链接期 `section .MEMORYx size exceeds max DM size 256KB` 错误）。

2. **显式物理 Bank 绑定修饰符 (`__fix0__` ~ `__fix5__`)**：
   - 语法：`__fix1__ v16fp32 buf[N];`
   - 语义：强制将变量起始地址绑定至对应 APE 的物理 Bank：
     - 物理映射关系由公式 `physical_dm_id = ape_id * 6 + fix_id` 确定；
     - `__fix0__` 映射为 `DM0 / DM6 / DM12 / DM18`；
     - `__fix1__` 映射为 `DM1 / DM7 / DM13 / DM19`；
     - `__fix2__` ~ `__fix5__` 分别映射为各 APE 的 4 个 ACC Bank。

---

## §3 Memory Cluster 硬件冲突机制分析

根据厂商库 `device_info.h` 硬件规范，目标 LPU 真正的硬件争用机制为 **Memory Cluster 冲突**：

1. **Cluster 归属与并发冲突规则**：
   - 单 APC 内的 24 个 Bank 组织在不同 Memory Cluster 中，规则为：`DM(i)` 与 `DM(i+12)` 属于同一个 Memory Cluster；
   - **并发访问互斥**：APE0 与 APE1 不能同时访问同一个 Memory Cluster（否则触发硬件 Stall）；APE2 与 APE3 亦不能同时访问同一个 Memory Cluster（触发硬件 Stall）；
   - **累加单元共享互斥**：同一 Memory Cluster 内部共享同一套累加单元（Accumulation Unit），因此两个 APE 严禁在同一时刻对同一 Cluster 发起写回存储操作（`Store(WB)`），否则引发硬件流水线 Stall。

2. **单核内部变量别名与多核冲突实验的区别**：
   - 先前在单核环境下尝试将权重 $W$ 与输入 $X$ 同时修饰为 `__fix1__`，在物理上使得两个数组首地址完全重合（内存重叠别名 Bug），DMA 载入时相互覆盖破坏，导致内核启动异常（`device_cycles: 0`），该现象属于地址冲突错误，并非硬件 Bank 争用测量；
   - 真正的 Memory Cluster 争用属于**多核跨 APE 并发访问冲突**（例如 APE0 访问 DM0 时 APE1 访问 DM12），无法在单核单线程中构造。相关跨核争用评测已归档至 `BLOCKERS.md` §8 供后续多核微基准展开。

---

## §4 消融实验分析与实测对比

消融实验基于单 ELF 二进制产物（`bank_ablation.out`，MD5: `8792b2ed03a7acc36c99c2eca52f10dd`，Commit: `840943b`），实测数据记录于 `results/bank_ablation.csv`。

### 1. 默认分配 vs 显式指定 (`__fix1~3__`) 对比

| 算子规模 | 配置类型 | 校验状态 | 硬件执行周期 (device_cycles) |
| :--- | :--- | :---: | :---: |
| **GEMM M=1** | 编译器默认 (`__mem0~2__`) | 通过 (`max_diff: 3.58e-7`) | **614,300** |
| | 显式指定 (`__fix1~3__`) | 通过 (`max_diff: 3.58e-7`) | **614,300** |
| **GEMM M=4** | 编译器默认 (`__mem0~2__`) | 通过 (`max_diff: 3.58e-7`) | **1,882,877** |
| | 显式指定 (`__fix1~3__`) | 通过 (`max_diff: 3.58e-7`) | **1,882,877** |
| **GEMM M=8** | 编译器默认 (`__mem0~2__`) | 通过 (`max_diff: 3.58e-7`) | **3,574,313** |
| | 显式指定 (`__fix1~3__`) | 通过 (`max_diff: 3.58e-7`) | **3,574,313** |

- **结论**：
  在各 $M$ 规模下，芯片硬件周期 `device_cycles` **逐位完全一致**（614,300、1,882,877、3,574,313）。
  由此表明：`scale-cc` 编译器在处理常规 `__mem__` 声明时已具备成熟的无冲突物理 Bank 映射能力，采用显式 `__fix__` 声明相比编译器默认分配**未观察到显著差异**。

### 2. 32 KB 建议 Region vs 256 KB 大 Region 开销对比

针对 Decode Attention v1 片上常驻算子，保持相同的计算负载（$L=128, d=64$），对比声明 32 KB 缓冲区（`attn_compact`）与声明 256 KB 缓冲区（`attn_large`）的开销：

| 算子配置 | Region 声明尺寸 | 实际计算量 | 硬件执行周期 (device_cycles) | 精度校验 |
| :--- | :---: | :---: | :---: | :---: |
| `attn_compact` | **32 KB** (`RESIDENT_L=128`) | $L=128$ | **347,657** | 通过 (`max_diff: 3.26e-9`) |
| `attn_large` | **256 KB** (`RESIDENT_L=1024`) | $L=128$ | **347,657** | 通过 (`max_diff: 3.26e-9`) |

- **结论**：
  硬件芯片执行周期 `device_cycles` 均为 **347,657**（完全一致）。声明 256 KB 大 Region 并未引入任何寻址、地址转换或物理 Bank 访问延迟惩罚。
  编程指南中“建议 Region $\le 32\text{ KB}$”的规则主要源于单 APC 内部 4 个 APE 共享或多 Warp 细粒度并发时的防溢出设计规范；在单 APE 独立运行或单 Warp 算子开发中，直接使用物理上限 256 KB Region 是完全安全且零额外性能开销的。两者性能**未观察到显著差异**。
