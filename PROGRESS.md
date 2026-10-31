# 工作进度日志 (PROGRESS.md)

## 2026-10-31
### 1. 环境验证与基线确认
- **宿主机**：Windows 11 WSL2 (Ubuntu 24.04)，开启 KVM 嵌套虚拟化与 Linger 常驻。
- **客体机 (QEMU CModel)**：通过 tmux 保持后台运行，Gem5 4 APC 仿真核心就绪，A76 固件就绪。
- **端口与连通性**：SSH 端口自动提取成功，宿主机通过 `scripts/guest_run.sh` 实现免密直连。
- **健康检查**：
  - `modprobe maps` 成功加载内核模块。
  - `/dev/dri/card1` 设备节点正常。
  - `umd_test/memory/memory_basic_test` 8 项测试全部 PASS。
- **官方 Demo 验证**：
  - 成功编译并执行 `umd_test/kernels` 完整 demo 工程（`a.out`），涵盖 `vaddu32`、`vmulu32`、`vsubu32`、`vtanhfp32` 等 kernel，计算结果校验全部通过（Test PASS）。

### 2. 下一步计划
- **阶段 0**：完成 `notes/demo_walkthrough.md` 与 `notes/mpu_capabilities.md`。
- **阶段 1**：调研并建立可信的仿真时间计时方法（`METHODOLOGY.md`），重点验证 Host Event、Device Cycle 计数器与 gem5 stats。
