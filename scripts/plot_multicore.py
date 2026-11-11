#!/usr/bin/env python3
"""
plot_multicore.py - Multi-core scaling and N vs. C mode visualization for LPU-Bench.
Generates figures/multicore_scaling.png from results/gemm_multicore.csv.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "gemm_multicore.csv")
OUT_FIG = os.path.join(PROJECT_ROOT, "figures", "multicore_scaling.png")

# Setup font configuration
import matplotlib
import matplotlib.font_manager as fm
font_path = '/mnt/c/Windows/Fonts/simhei.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'Helvetica']
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.edgecolor'] = '#cccccc'
matplotlib.rcParams['axes.linewidth'] = 0.8

def main():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        print(f"[ERROR] CSV not found or empty: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Ensure numeric types
    numeric_cols = ['num_threads', 'M', 'K', 'N', 'sim_cycles', 'device_cycles', 'flops', 'flops_per_cycle', 'total_bytes', 'bytes_per_cycle']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Aggregate by (exec_mode, num_threads, M)
    agg = df.groupby(['exec_mode', 'num_threads', 'M']).agg(
        dev_cycles_mean=('device_cycles', 'mean'),
        dev_cycles_std=('device_cycles', 'std'),
        flops_per_cycle_mean=('flops_per_cycle', 'mean'),
        bytes_per_cycle_mean=('bytes_per_cycle', 'mean')
    ).reset_index()

    # Calculate speedup relative to N mode threads=1 for each M using hardware device_cycles
    base_n1 = agg[(agg['exec_mode'] == 'N') & (agg['num_threads'] == 1)].set_index('M')['dev_cycles_mean'].to_dict()
    agg['speedup'] = agg.apply(lambda r: base_n1.get(r['M'], np.nan) / r['dev_cycles_mean'] if r['M'] in base_n1 else np.nan, axis=1)

    print("\n--- Summary Table ---")
    print(agg[['exec_mode', 'num_threads', 'M', 'dev_cycles_mean', 'speedup', 'flops_per_cycle_mean']])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 8: '#d62728'}
    markers = {1: 'o', 2: 's', 4: '^', 8: 'D'}

    # ------------------ Subplot 1: Speedup vs Threads ------------------
    threads_list = [1, 2, 4]
    ax1.plot(threads_list, threads_list, 'k--', label='理想线性加速 (Ideal Linear, y=x)', alpha=0.6, linewidth=1.5)

    for m in [1, 2, 4, 8]:
        sub_n = agg[(agg['exec_mode'] == 'N') & (agg['M'] == m)].sort_values('num_threads')
        if not sub_n.empty:
            ax1.plot(sub_n['num_threads'], sub_n['speedup'],
                     color=colors[m], marker=markers[m], markersize=8, linewidth=2,
                     label=f'N 模式 M={m}')
            for _, row in sub_n.iterrows():
                ax1.annotate(f"{row['speedup']:.2f}×",
                             (row['num_threads'], row['speedup']),
                             textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, color=colors[m])

        sub_c = agg[(agg['exec_mode'] == 'C') & (agg['M'] == m)]
        if not sub_c.empty:
            for _, row in sub_c.iterrows():
                ax1.plot(row['num_threads'], row['speedup'],
                         color=colors[m], marker='*', markersize=12,
                         markeredgecolor='black', markeredgewidth=0.8,
                         linestyle='None', label=f'C 模式 M={m} (4核独占)' if m == 1 else None)
                ax1.annotate(f"C={row['speedup']:.2f}×",
                             (row['num_threads'], row['speedup']),
                             textcoords="offset points", xytext=(8, -10), ha='left', fontsize=8,
                             fontweight='bold', color=colors[m])

    ax1.set_xlabel('线程数 / 核数 (Threads / APEs)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('硬件加速比 (Speedup based on device_cycles)', fontsize=11, fontweight='bold')
    ax1.set_title('单 APC 内多核扩展加速比 (GEMM K=256, N=256)', fontsize=12, fontweight='bold')
    ax1.set_xticks([1, 2, 4])
    ax1.set_xlim(0.8, 4.4)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # ------------------ Subplot 2: Throughput (flops/cycle) ------------------
    for m in [1, 2, 4, 8]:
        sub_n = agg[(agg['exec_mode'] == 'N') & (agg['M'] == m)].sort_values('num_threads')
        if not sub_n.empty:
            ax2.plot(sub_n['num_threads'], sub_n['flops_per_cycle_mean'],
                     color=colors[m], marker=markers[m], markersize=8, linewidth=2,
                     label=f'N 模式 M={m}')

        sub_c = agg[(agg['exec_mode'] == 'C') & (agg['M'] == m)]
        if not sub_c.empty:
            for _, row in sub_c.iterrows():
                ax2.plot(row['num_threads'], row['flops_per_cycle_mean'],
                         color=colors[m], marker='*', markersize=12,
                         markeredgecolor='black', markeredgewidth=0.8,
                         linestyle='None', label=f'C 模式 M={m}' if m == 1 else None)

    ax2.set_xlabel('线程数 / 核数 (Threads / APEs)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('整核吞吐 (FLOP / cycle)', fontsize=11, fontweight='bold')
    ax2.set_title('多核聚合算力吞吐 (FLOP/cycle vs. Threads)', fontsize=12, fontweight='bold')
    ax2.set_xticks([1, 2, 4])
    ax2.set_xlim(0.8, 4.4)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)

    # Add caveat note banner across bottom
    caveat_text = (
        "【关键约束与物理说明】实测 DDR 带宽利用率仅达峰值带宽约 0.062%，多核聚合带宽随线程数近似线性增长系因未进入内存总线饱和区（零争用），\n"
        "绝不构成多核独占独立带宽假设成立的物理证据；所有测量基于单一构建产物 (MD5: bbf153e9b289b92bcf8055794e812953)，无交叉构建漂移。"
    )
    plt.figtext(0.5, 0.01, caveat_text, ha='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', edgecolor='#ffeeba', alpha=0.9),
                 color='#856404')

    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, bbox_inches='tight')
    plt.close()
    print(f"\n[SUCCESS] Figure generated and saved to: {OUT_FIG}")

if __name__ == "__main__":
    main()
