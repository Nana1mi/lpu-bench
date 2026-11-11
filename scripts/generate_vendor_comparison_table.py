#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

def generate_table():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gemm_csv = os.path.join(repo_root, "results", "gemm.csv")
    vendor_real_csv = os.path.join(repo_root, "results", "vendor_gemm_real.csv")
    vendor_old_csv = os.path.join(repo_root, "results", "vendor_gemm.csv")

    df_gemm = pd.read_csv(gemm_csv)
    # Filter non-empty build_id and version v0
    df_gemm = df_gemm[df_gemm['build_id'].fillna('').astype(str).str.strip() != '']
    df_v0 = df_gemm[df_gemm['version'] == 'v0']
    naive_agg = df_v0.groupby('M')[['device_cycles', 'flops_per_cycle']].mean().reset_index()

    df_vendor_real = pd.read_csv(vendor_real_csv)
    df_vendor_real = df_vendor_real[df_vendor_real['validity'] == 'valid']
    vendor_real_agg = df_vendor_real.groupby(['dtype', 'M'])[['device_cycles', 'flops_per_cycle']].mean().reset_index()

    df_vendor_old = pd.read_csv(vendor_old_csv)
    vendor_old_agg = df_vendor_old.groupby(['dtype', 'M'])[['device_cycles', 'flops_per_cycle']].mean().reset_index()

    naive_map = {}
    for _, row in naive_agg.iterrows():
        naive_map[int(row['M'])] = {
            'cycles': row['device_cycles'],
            'flops_per_cycle': row['flops_per_cycle']
        }

    lines = []
    lines.append("| 算子实现层级 | 数据精度 (dtype) | 批大小 $M$ | 硬件执行周期 (device_cycles) | 计算吞吐 (FLOPs/cycle) | 相对单 APC 峰值 (1024) / 所占核峰值利用率 | 相对 Naive C 算子加速比 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    # 1. Naive C (1 APE)
    for m in [1, 2, 4, 8]:
        c = naive_map[m]['cycles']
        fpc = naive_map[m]['flops_per_cycle']
        util_ape = (fpc / 256.0) * 100.0
        util_apc = (fpc / 1024.0) * 100.0
        c_str = f"{int(round(c)):,}"
        lines.append(f"| **自研 Naive C (`gemm.sc`, 单 APE)** | FP32 | $M={m}$ | {c_str} | {fpc:.4f} (单 APE) | {util_ape:.3f}% (对单 APE 峰值 256; 对单 APC 1024 为 {util_apc:.3f}%) | **1.00×** (基线) |")

    # 2. Vendor FP32 (Real libmablasLt.so via 00_basic_matmul, 1 APC = 4 APEs)
    v_fp32 = vendor_real_agg[vendor_real_agg['dtype'] == 'fp32'].sort_values('M')
    for _, row in v_fp32.iterrows():
        m = int(row['M'])
        c = row['device_cycles']
        fpc = row['flops_per_cycle']
        per_ape_fpc = fpc / 4.0
        util_apc = (fpc / 1024.0) * 100.0
        c_str = f"{int(round(c)):,}"
        if m in naive_map:
            e2e_speedup = naive_map[m]['cycles'] / c
            per_ape_speedup = e2e_speedup / 4.0
            speedup_str = f"**{per_ape_speedup:.2f}×** (单 APE 归一化) / **{e2e_speedup:.2f}×** (含 4× 硬件规模差异端到端)"
        else:
            speedup_str = f"— (无 Naive C M={m} 实测基线)"
        lines.append(f"| **厂商 `mablaslt` (`00_basic_matmul` 真实库, 1 APC=4 APE)** | FP32 | $M={m}$ | {c_str} | {fpc:.3f} (4-APE 聚合; 单 APE {per_ape_fpc:.3f}) | **{util_apc:.3f}%** (基准 1024 FLOPs/cyc) | {speedup_str} |")

    # 3. Vendor FP16 (Still pending re-measurement from vendor_gemm.csv, 1 APC = 4 APEs)
    v_fp16 = vendor_old_agg[vendor_old_agg['dtype'] == 'fp16'].sort_values('M')
    for _, row in v_fp16.iterrows():
        m = int(row['M'])
        c = row['device_cycles']
        fpc = row['flops_per_cycle']
        util_apc = (fpc / 8192.0) * 100.0
        c_str = f"{int(round(c)):,}"
        lines.append(f"| **厂商 `mablaslt` (`slice_k` 模式, 1 APC=4 APE)* [待重新测量]** | FP16 | $M={m}$ | {c_str} | {fpc:.3f} (4-APE 聚合) | {util_apc:.3f}% (基准 8192 FLOPs/cyc) | — (跨精度不直接对比) |")

    # 4. Theoretical Peaks
    lines.append("| **理论物理峰值 (FP32)** | FP32 | 任意 | — | **256.00 (单 APE) / 1024.00 (单 APC)** | **100.0%** | **878×** (单 APE 峰值 vs Naive C M=8) |")
    lines.append("| **理论物理峰值 (FP16)** | FP16 | 任意 | — | **2048.00 (单 APE) / 8192.00 (单 APC)** | **100.0%** | — (跨精度不直接对比) |")

    table_md = "\n".join(lines)
    return table_md

if __name__ == "__main__":
    table = generate_table()
    print(table)
