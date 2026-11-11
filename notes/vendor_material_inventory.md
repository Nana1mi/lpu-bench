# 芯片厂商涉密物料本地清单 (vendor_material_inventory.md)

本文档归档本地环境中保有的芯片厂商专有物料、具体来源以及不纳入公共版本控制的合规理由。供后续迁移机器、环境复现或人员交接时核对，避免物料遗失或意外泄露。

---

## 1. 涉密物料清单与来源明细

| 物料路径 / 范围 | 具体包含文件 | 来源渠道 | 为何不纳入版本控制 |
|---|---|---|---|
| `docs/` | `SCALE_Programming_Guide`、`maps-umd包使用说明`、`Installation_Document_2.0.x`、`搭建cmodel环境`、`SCALE_CModel_Installation_Summary` 等 | 人类研究员提供的厂商内部技术规范与 SDK 安装指南 | 属于芯片厂商内部未公开文档，包含底层寄存器映射、MPU 微架构指令集及驱动接口规范，`GOAL.md` 规则 6 明确要求禁止公开上传 |
| `kernels/official_demo/` | 厂商端到端示例源码（`main.sc`、`kernels.sc`、`perf.sc`、`end2end.sc`、`apc_perf_test.sc` 等）、头文件（`maops.h`）、MPU 汇编（`mpu_add_u32.m.asm`、`mpu_tanh_fp32_a.m.asm`、`mpu_tanh_fp32_s.m.asm`）、测试数据与驱动执行日志（`malog_2896.txt`、`scalehost_log_2896.txt`） | Programming Guide 5.2.3 节配套的 SCALE 官方端到端示例工程 | 包含厂商专有 MPU 向量核手写 VLIW 微码汇编、驱动底层接口宏以及包含真实环境路径与进程信息的运行日志 |
| `kernels/test_fma/` | `mpu_add_u32.m.asm` | 从 `kernels/official_demo/` 复制，用于对比编译 C 与手写汇编性能基准 | 厂商专有 MPU 汇编实现，保留本地用于基准微测试编译，不再纳入 git 跟踪 |
| `results/` | `maps_event_driver_log_excerpt.txt` | QEMU 客体机驱动调试日志抽取片段 | 包含驱动内部未公开函数实现调用栈与调试信息 |
| 宿主机临时物料区（仓库外） | `/mnt/e/demo4scalee2e.tgz`、解压至 `/home/rog/lpu-workspace/vendor_demo/` | 厂商交付的 SCALE 端到端官方算子 Demo 包（含 MaBLASLt、MaOps、FlashAttention、多核 DSHM 示例等 37 个模块） | 包含厂商专有手写汇编算子（`mpu_asm/*.m.asm`）、FlashAttention 移植实现、驱动级底层链接脚本（`apc-fa-cmode.ld`），属于核心涉密源码，严禁提交至版本控制 |
| 宿主机根目录驱动区（仓库外） | `/home/rog/lpu-workspace/scale/driver/` 下全部驱动与模拟器镜像（`pciemu`、`mapkg`、客体机 QEMU 镜像） | 厂商交付的二进制驱动包与 gem5 CModel 运行环境 | 二进制闭源物料，体积巨大且包含固件代码，位于仓库目录树之外 |

---

## 2. 本地保留与环境复现说明

上述物料在本地文件系统中均予以**完整保留**，确保当前工程的构建、运行与实验比对不受任何影响：
- 编译依赖：`kernels/test_fma` 在本地仍可直接引用本地汇编文件执行编译；
- 文档查阅：`docs/` 下规范继续作为算法实现与硬件约束（APC/Bank/Region）的本地指导依据；
- 换机迁移：若需在其他服务器上复现本仓库，需由人类研究员通过私有离线方式重新分发上述 `docs/` 与 `kernels/official_demo/` 目录。
