# Aurora MPU 向量能力与编程特性分析 (MPU Capabilities)

## 1. 硬件架构特征

Aurora 芯片计算单元称为 **APE (Aurora Processing Element)**，由独立的 **SPU** 与 **MPU** 深度配合构成。

| 模块 | 核心定位 | 寄存器与数据位宽 | 访存能力 |
|---|---|---|---|
| **SPU** | 标量乱序执行、DMA 驱动、任务同步调度 | 32/64 位通用标量寄存器 | 可直接通过虚拟/物理地址访问 DDR 及片上 DM |
| **MPU** | VLIW 向量超长指令字计算核心 | 512 位宽向量寄存器 (512-bit SIMD) | **仅能访问片上 DM**，无 DDR 直接访存权 |

- **APC (Aurora Processing Cluster)**：每个 APC 包含 4 个 APE，共享由 24 个独立的 Bank 组成的片上高速 SRAM（Data Memory, DM）。
- **SIMD vs. SIMT**：与 GPU 的 SIMT（32 线程 Warp 隐式并行）不同，Aurora 是典型的 **SIMD** 架构（`warpSize == 1`），单线程即可直接控制 512 位向量指令。

---

## 2. 向量数据类型定义 (`aurora.h`)

Aurora 工具链在 `llvm/aurora.h` 中内置了丰富的 512 位及子位宽向量类型定义，对应 GCC/Clang `__attribute__((__vector_size__(64)))` 属性：

### 2.1 基础向量类型
- **单精度浮点 (FP32)**：
  - `v16fp32`: 64 字节，包含 16 个 32-bit 单精度浮点数（LLM GEMM 及 Attention 的标准高保真类型）。
- **半精度与 BFloat16**：
  - `v32fp16`: 64 字节，包含 32 个 16-bit 半精度浮点数。
  - `v32bf16`: 64 字节，包含 32 个 16-bit BFloat16 浮点数。
- **低位宽整型 / 量化数据类型**：
  - `v64s8` / `v64u8` / `v64i8`: 64 字节，包含 64 个 8-bit 整型数（INT8 量化推断专用）。
  - `v64fp8e4` / `v64fp8e5`: 针对 FP8 (E4M3 / E5M2) 的向量存储。
- **定点与 32 位整型**：
  - `v16u32` / `v16s32`: 64 字节，包含 16 个 32-bit 无符号/有符号整型。

### 2.2 扩展向量寄存器
- `v32s32` / `v32fp32`: 1024 位宽，包含 32 个 32-bit 元素（通常用于 FMA 累加器 MR 寄存器）。

---

## 3. SPU 调用 MPU 的三种范式

根据 SCALE 编程手册规范，SPU 控制 MPU 有三种模式：

```
                +---------------------------------------+
                |           SPU 调度线程                |
                +---------------------------------------+
                     |              |               |
             1. async_call    2. sync_call    3. async_enqueue
                     v              v               v
                +---------------------------------------+
                |           MPU 计算核心                |
                +---------------------------------------+
```

1. **`async_call` (默认/隐式异步调用)**：
   - SPU 调用带有 `KERNEL_M` 修饰的函数。
   - SPU 不等待立即返回，可继续执行自身代码（如准备下一轮 DMA 搬运）。
   - 当 SPU 需要 MPU 结果时，调用 `MPU_SYNC()` 或屏障等待。
2. **`sync_call` (同步调用)**：
   - SPU 阻塞等待直到 MPU 完成当前向量计算并返回。
   - 适用于简短的控制依赖计算。
3. **`async_enqueue` (异步硬件任务队列)**：
   - SPU 将计算任务推入硬件任务队列（Ring Buffer），避免 SPU-MPU 握手开销。
   - 是实现**乒乓双缓冲 (Double Buffering) 与 DMA 流水完全重叠**的终极武器。
   - 注意：手册规定 `async_enqueue` 与前两类调用不可在同一 Kernel 内混用。

---

## 4. 关键 Intrinsics 与高级计算原语

Aurora 提供了两套 MPU 接口：
1. **MPU C/C++ (`KERNEL_M`) 算子重载**：
   直接支持标准 C++ 运算符重载：
   ```cpp
   v16fp32 c = a + b;
   v16fp32 d = a * b;
   ```
2. **底层硬件加速 Intrinsics (`aurora_m-cxx-intrin.h`)**：
   - **`scale::__vclear_mr()`**: 清空专用硬件 FMA 累加寄存器 (`FMAMR`)。
   - **`scale::__vfmadd_mr(FMAMR, src1, src2)`**: 硬件级乘加融合指令（Fused Multiply-Add, FMA），单周期完成 16 路 FP32 乘累加计算。
   - **`scale::__vradd16(src)`**: 16 路向量水平归约求和（Vector Reduction Add），用于点积最终求标量和。
   - **`scale::__vextract_lo / __vextract_hi`**: 向量高低半分拆。

---

## 5. 对 LLM 典型算子的指导意义

### 5.1 FFN / GEMM 算子
- **特征**：计算密集型 ($O(N^3)$ 计算对 $O(N^2)$ 搬运)。
- **映射策略**：
  - 矩阵分块放置在片上 DM（如分块 $16 \times 16$）。
  - MPU 运行 `v16fp32` 向量 FMA 外积或内积循环。
  - SPU 采用双缓冲预取下一分块，隐藏 DDR 访存延迟。

### 5.2 Decode Attention 算子
- **特征**：极度访存受限（Memory-Bound，$O(N)$ 计算对 $O(N)$ 搬运）。
- **映射策略**：
  - 在 Batch=1 的 Decode 阶段，Query 向量极小（如 $1 \times 64$），每次推理步都需要读取全部历史 KV Cache。
  - **DDR 模式 (v0)**：每次从 DDR 搬运 KV Cache，受限于 DDR 带宽。
  - **片上 DM 常驻模式 (v1)**：若将前缀或关键 KV Cache 留存在片上 DM（利用 Aurora 多 Bank DM 大容量），MPU 直接在片上读写，性能将实现数量级飞跃。
  - **流水重叠 (v2)**：使用 SPU 边搬运后续 KV Tile，边计算当前 Tile 的 $Q \cdot K^T$ 点积。
