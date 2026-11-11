# 服务器 17 CModel 交叉验证基准备忘录 (cross_validation_server17.md)

本文档归档课题组另一成员在**服务器 17**上部署的同套 LPU CModel 仿真环境与当前开发机（WSL 环境）进行交叉比对的基准配置、待交付二进制校验码与比对目标。

---

## §1 交叉验证目标与核心意义

1. **环境等价性验证**：
   排除宿主机虚拟化（Windows 11 + WSL2 Ubuntu）对底层 gem5 指令级仿真与 MAPS 驱动结果的可能干扰，确认硬件仿真周期的确定性。
2. **CModel 配置一致性前置审查**：
   课题组正在研发基于 Frontier 的系统级模拟器，当前算子基准数据将作为底层执行参数输入。若两台服务器的 CModel 配置文件（`config.ini`）存在参数偏置（如 SimpleMemory 延迟、APC 数量或总线频率），将导致混合输入数据失真。

---

## §2 交付二进制物料与基准指标

为确保比对完全遵循“相同 `build_id`”原则，比对使用**已编译并固化的单一二进制文件**（无需在服务器 17 上重复编译源码）：

| 验证项 | 目标测试程序 | 二进制 MD5 校验码 | 运行命令 | 本机标准周期数 (device_cycles) |
| :--- | :--- | :---: | :--- | :---: |
| **微基准 (Timer)** | `test_timer.out` | `d81ca9cb110eb7efa1affbfc3e3453a4` | `./test_timer.out 1000` | **1,000** |
| **单核 GEMM** | `gemm.out` | `7aa34dfcf47bb4d06513c651f980a002` | `./gemm.out --M 1 --repeats 1` | **648,903** |
| **多核 GEMM** | `gemm_multicore.out` | `bbf153e9b289b92bcf8055794e812953` | `./gemm_multicore.out --mode N --threads 1 --M 1 --repeats 1` | **648,903** |
| | | | `./gemm_multicore.out --mode N --threads 4 --M 1 --repeats 1` | **351,150** |
| | | | `./gemm_multicore.out --mode C --threads 4 --M 1 --repeats 1` | **351,159** |
| **Bank 消融** | `bank_ablation.out` | `8792b2ed03a7acc36c99c2eca52f10dd` | `./bank_ablation.out --test gemm_default --M 1 --repeats 1` | **614,300** |

---

## §3 结果判据与差异排查清单

### 1. 结果判据
- **逐位一致 (Bit-for-Bit Identical)**：若服务器 17 实测 `device_cycles` 与上述基准完全一致，则表明两套仿真环境完全等价，数据可直接合并进入 Frontier 系统级模拟器。
- **存在差异 (Discrepant)**：若出现周期数分歧，说明两套环境下的 CModel 仿真配置存在底层参数分歧，**严禁合并两边数据**，必须立即排查配置。

### 2. `config.ini` 参数对齐排查清单
若结果不一致，比对两台机器上的 `driver/pciemu/log/gem5/device1/config.ini`：
1. `system.m_apc` 节点数量（确认是否均为 4 APC）；
2. `SimpleMemory` 配置：`latency`、`latency_var`、`bandwidth` 是否一致；
3. `system.m_apc*.pipeline` 标志（确认是否均为 `false`）；
4. `system.clk_domain.clock` 仿真时钟频率设置；
5. `MAPS-RUNTIME` 与 KMD 驱动版本（确认是否均为 `1.2.3-HEAD-8b4ab0d`）。

---

## §4 当前协调状态

- **状态**：二进制文件已就绪并归档 MD5。目前正等待服务器 17 负责同学协调运行环境并反馈执行结果。
- **依赖管理**：已同步登记于 `BLOCKERS.md` §3，不阻塞本机后续分析任务。
