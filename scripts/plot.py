#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
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

def read_measured_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'build_id' in df.columns and df['build_id'].fillna('').astype(str).str.strip().ne('').any():
        df = df[df['build_id'].fillna('').astype(str).str.strip() != '']
    return df

def plot_k1_gemm():
    csv_path = os.path.join(RESULTS_DIR, "gemm.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} does not exist, skipping K1 plot")
        return

    df = read_measured_csv(csv_path)
    # Average across repeats
    agg = df.groupby(['version', 'M'])[['device_cycles', 'flops_per_cycle']].mean().reset_index()

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    v0_data = agg[agg['version'] == 'v0'].sort_values('M')
    v1_data = agg[agg['version'] == 'v1'].sort_values('M')

    color_v0 = '#1f77b4'
    color_v1 = '#ff7f0e'

    ax1.set_xlabel('Token / Batch Size M (K=256, N=256)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Effective Compute Throughput (FLOPs / Cycle)', fontsize=12, fontweight='bold')

    l1 = ax1.plot(v0_data['M'], v0_data['flops_per_cycle'], marker='o', linewidth=2.2, color=color_v0, label='v0 Serial (DMA + Compute)')
    l2 = ax1.plot(v1_data['M'], v1_data['flops_per_cycle'], marker='s', linewidth=2.2, color=color_v1, label='v1 Overlap (Double-Buffered DMA)')

    ax1.set_xticks(v0_data['M'])
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Secondary axis for total cycles
    ax2 = ax1.twinx()
    ax2.set_ylabel('Total Device Cycles (M-Cycles)', fontsize=12, color='#555555')
    l3 = ax2.plot(v0_data['M'], v0_data['device_cycles'] / 1e6, marker='^', linestyle=':', color=color_v0, alpha=0.5, label='v0 Cycles (M)')
    l4 = ax2.plot(v1_data['M'], v1_data['device_cycles'] / 1e6, marker='v', linestyle=':', color=color_v1, alpha=0.5, label='v1 Cycles (M)')
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax2.set_ylim(bottom=0)
    ax2.grid(False)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9)

    plt.title('K1: LPU GEMM Compute Throughput vs. Batch/Token Scale M', fontsize=14, fontweight='bold', pad=15)
    out_file = os.path.join(FIGURES_DIR, "k1_gemm_utilization.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[PLOT] Generated {out_file}")

def plot_k2_attn():
    csv_path = os.path.join(RESULTS_DIR, "decode_attn.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} does not exist, skipping K2 plot")
        return

    df = read_measured_csv(csv_path)
    agg = df.groupby(['version', 'L'])[['device_cycles', 'effective_bw_bytes_per_cycle']].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    colors = {'v0': '#d62728', 'v1': '#2ca02c', 'v2': '#1f77b4'}
    markers = {'v0': 'o', 'v1': '*', 'v2': 's'}
    labels = {
        'v0': 'v0 DDR Stream (Serial)',
        'v1': 'v1 On-Chip DM Resident (SRAM)',
        'v2': 'v2 DDR Stream (Double-Buffered Overlap)'
    }

    for ver in ['v0', 'v2', 'v1']:
        sub = agg[agg['version'] == ver].sort_values('L')
        if not sub.empty:
            ax.plot(sub['L'], sub['device_cycles'] / 1e3,
                    marker=markers[ver], markersize=8 if ver != 'v1' else 11,
                    linewidth=2.2, color=colors[ver], label=labels[ver])

    ax.set_xlabel('Context Sequence Length L (tokens, head_dim d=64)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Latency (Thousand Cycles)', fontsize=12, fontweight='bold')
    ax.set_title('K2: LLM Decode Attention Latency vs. KV Cache Locality', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks([64, 128, 256, 512, 1024])
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    out_file = os.path.join(FIGURES_DIR, "k2_attn_cycles.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[PLOT] Generated {out_file}")

def plot_k2_breakdown():
    csv_path = os.path.join(RESULTS_DIR, "decode_attn.csv")
    if not os.path.exists(csv_path):
        return

    df = read_measured_csv(csv_path)
    agg = df.groupby(['version', 'L'])['device_cycles'].mean().unstack(level=0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    lengths = [64, 128, 256, 512, 1024]
    valid_lengths = [l for l in lengths if l in agg.index and 'v1' in agg.columns and 'v0' in agg.columns]

    if valid_lengths:
        compute_cycles = [agg.loc[l, 'v1'] / 1e3 for l in valid_lengths]
        dma_overhead = [(agg.loc[l, 'v0'] - agg.loc[l, 'v1']) / 1e3 for l in valid_lengths]

        x = np.arange(len(valid_lengths))
        width = 0.45

        p1 = ax.bar(x, compute_cycles, width, label='MPU Compute + SRAM Access (Resident DM)', color='#2ca02c', alpha=0.85)
        p2 = ax.bar(x, dma_overhead, width, bottom=compute_cycles, label='DDR SDMA Fetch Overhead', color='#d62728', alpha=0.85)

        ax.set_ylabel('Cycles (Thousands)', fontsize=12, fontweight='bold')
        ax.set_title('Decode Attention Execution Time Breakdown (L=64 to 1024)', fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([f'L={l}' for l in valid_lengths], fontsize=11)
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')

        out_file = os.path.join(FIGURES_DIR, "k2_breakdown.png")
        fig.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)
        print(f"[PLOT] Generated {out_file}")

def plot_k3_speedup():
    gemm_path = os.path.join(RESULTS_DIR, "gemm.csv")
    attn_path = os.path.join(RESULTS_DIR, "decode_attn.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    if os.path.exists(gemm_path):
        df_g = read_measured_csv(gemm_path)
        g_agg = df_g.groupby(['version', 'M'])['device_cycles'].mean().unstack(level=0)
        if 'v0' in g_agg.columns and 'v1' in g_agg.columns:
            g_speedup = g_agg['v0'] / g_agg['v1']
            bars1 = ax1.bar([str(m) for m in g_speedup.index], g_speedup.values, color='#ff7f0e', alpha=0.85, edgecolor='#cc6600')
            ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1)
            ax1.set_xlabel('Matrix Dimension M (K=N=256)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Speedup (v0_serial / v1_overlap)', fontsize=11, fontweight='bold')
            ax1.set_title('GEMM Double-Buffering Speedup vs M', fontsize=12, fontweight='bold')
            ax1.set_ylim(0.8, max(1.5, max(g_speedup.values) * 1.15))
            ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}x', ha='center', va='bottom', fontsize=8)

    if os.path.exists(attn_path):
        df_a = read_measured_csv(attn_path)
        a_agg = df_a.groupby(['version', 'L'])['device_cycles'].mean().unstack(level=0)
        if 'v0' in a_agg.columns and 'v1' in a_agg.columns:
            resident_speedup = (a_agg['v0'] / a_agg['v1']).dropna()
            overlap_speedup = (a_agg['v0'] / a_agg['v2']).dropna() if 'v2' in a_agg.columns else None

            labels = [f"L={l}" for l in resident_speedup.index]
            x = np.arange(len(labels))
            w = 0.35

            b1 = ax2.bar(x - w/2, resident_speedup.values, w, label='v1 DM Resident vs. v0 DDR', color='#2ca02c', alpha=0.85)
            if overlap_speedup is not None:
                b2 = ax2.bar(x + w/2, overlap_speedup.loc[resident_speedup.index].values, w, label='v2 Overlap vs. v0 DDR', color='#1f77b4', alpha=0.85)

            ax2.axhline(1.0, color='gray', linestyle='--', linewidth=1)
            ax2.set_xlabel('Sequence Length L', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Speedup vs. v0 Baseline', fontsize=11, fontweight='bold')
            ax2.set_title('Attention Speedup: On-Chip SRAM vs. DDR Streaming', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylim(0.8, max(1.5, max(resident_speedup.values) * 1.2))
            ax2.legend(loc='upper right', frameon=True)
            ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
            for bar in b1:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}x', ha='center', va='bottom', fontsize=8)

    out_file = os.path.join(FIGURES_DIR, "k3_speedup.png")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    print(f"[PLOT] Generated {out_file}")

def clean_log_format(x, pos):
    if x <= 0:
        return ""
    lx = np.log10(x)
    if np.isclose(lx, np.round(lx)):
        return f"$10^{{{int(np.round(lx))}}}$"
    return f"{x:g}"

def plot_roofline():
    gemm_path = os.path.join(RESULTS_DIR, "gemm.csv")
    attn_path = os.path.join(RESULTS_DIR, "decode_attn.csv")
    multicore_path = os.path.join(RESULTS_DIR, "gemm_multicore.csv")
    vendor_gemm_path = os.path.join(RESULTS_DIR, "vendor_gemm.csv")

    # Canvas ratio: ~4:3 with figsize=(13.0, 9.5), subplots_adjust right=0.70
    fig, ax = plt.subplots(figsize=(13.0, 9.5), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.70, top=0.93, bottom=0.08)

    # 1. Hardware Specifications (Vendor official from device_info.h / matmul_perf_info.h)
    freq_ghz = 1.6   # 1.6 GHz nominal frequency
    num_apes = 128   # 128 APEs per die (32 APCs * 4 APEs)

    # Peak Bandwidth
    peak_bw_die = 3276.8              # GB/s (Die peak GDDR6)
    peak_bw_ape = peak_bw_die / num_apes  # 25.6 GB/s (16 Bytes/cycle @ 1.6 GHz)

    # Peak Compute Throughput (Three Precisions, Single Die / Single APE Amortized)
    # INT8: 840 TFLOPS die / 6562.5 GFLOPS single APE (4096 FLOPs/cycle)
    peak_flops_int8_die = 840000.0
    peak_flops_int8_ape = peak_flops_int8_die / num_apes  # 6562.5 GFLOPS
    r_int8 = peak_flops_int8_die / peak_bw_die             # 256.35 FLOP/byte

    # FP16: 420 TFLOPS die / 3281.25 GFLOPS single APE (2048 FLOPs/cycle)
    peak_flops_fp16_die = 420000.0
    peak_flops_fp16_ape = peak_flops_fp16_die / num_apes  # 3281.25 GFLOPS
    r_fp16 = peak_flops_fp16_die / peak_bw_die             # 128.17 FLOP/byte

    # FP32: 52.5 TFLOPS die / 409.6 GFLOPS single APE (256 FLOPs/cycle)
    peak_flops_fp32_die = 52500.0
    peak_flops_fp32_ape = peak_flops_fp32_die / num_apes  # 409.6 GFLOPS
    r_fp32 = peak_flops_fp32_die / peak_bw_die             # 16.02 FLOP/byte

    # X range for intensity (cover from 0.05 up to 600 FLOP/byte to show all 3 knees)
    i_min = 0.05
    i_max = 600.0
    i_vals = np.logspace(np.log10(i_min), np.log10(i_max), 600)

    # Roofline curves (Single APE amortized scale)
    roof_int8 = np.minimum(peak_bw_ape * i_vals, peak_flops_int8_ape)
    roof_fp16 = np.minimum(peak_bw_ape * i_vals, peak_flops_fp16_ape)
    roof_fp32 = np.minimum(peak_bw_ape * i_vals, peak_flops_fp32_ape)

    # Plot roofs
    ax.plot(i_vals, roof_int8, color='#7f2704', linestyle='-', linewidth=2.0,
            label=f'INT8 Roof: 840 TFLOPS (APE 6562.5 GFLOPS, R={r_int8:.1f} FLOP/B)')
    ax.plot(i_vals, roof_fp16, color='#d94801', linestyle='-', linewidth=2.0,
            label=f'FP16 Roof: 420 TFLOPS (APE 3281.25 GFLOPS, R={r_fp16:.1f} FLOP/B)')
    ax.plot(i_vals, roof_fp32, color='#08519c', linestyle='-', linewidth=2.0,
            label=f'FP32 Roof: 52.5 TFLOPS (APE 409.6 GFLOPS, R={r_fp32:.1f} FLOP/B)')

    # Bandwidth slope text
    ax.text(0.10, peak_bw_ape * 0.10 * 1.55,
            'Amortized Bandwidth: 25.6 GB/s (Die 3276.8 GB/s, 16 B/cycle)',
            rotation=36, fontsize=8.5, fontweight='bold', color='#333333')

    # Knee points scatter & annotations
    ax.scatter([r_int8], [peak_flops_int8_ape], color='#7f2704', s=60, zorder=5)
    ax.annotate(f'INT8 Knee\nR={r_int8:.1f} FLOP/B\n(840 TFLOPS)',
                xy=(r_int8, peak_flops_int8_ape), xytext=(r_int8 * 0.38, peak_flops_int8_ape * 1.25),
                arrowprops=dict(facecolor='#7f2704', arrowstyle='->', lw=1.2),
                fontsize=8.0, fontweight='bold', color='#7f2704')

    ax.scatter([r_fp16], [peak_flops_fp16_ape], color='#d94801', s=60, zorder=5)
    ax.annotate(f'FP16 Knee\nR={r_fp16:.1f} FLOP/B\n(420 TFLOPS)',
                xy=(r_fp16, peak_flops_fp16_ape), xytext=(r_fp16 * 0.35, peak_flops_fp16_ape * 0.45),
                arrowprops=dict(facecolor='#d94801', arrowstyle='->', lw=1.2),
                fontsize=8.0, fontweight='bold', color='#d94801')

    ax.scatter([r_fp32], [peak_flops_fp32_ape], color='#08519c', s=60, zorder=5)
    ax.annotate(f'FP32 Knee\nR={r_fp32:.1f} FLOP/B\n(52.5 TFLOPS)',
                xy=(r_fp32, peak_flops_fp32_ape), xytext=(r_fp32 * 1.35, peak_flops_fp32_ape * 0.45),
                arrowprops=dict(facecolor='#08519c', arrowstyle='->', lw=1.2),
                fontsize=8.0, fontweight='bold', color='#08519c')

    # Vertical reference lines
    ax.axvline(0.5, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(0.5 * 1.08, 0.08, 'Decode Attention\nI ≈ 0.5 FLOP/B (Fixed)\n(远低于全精度拐点)',
            color='purple', fontsize=8.0, fontweight='bold')

    # 2. Empirical Measured Data Points strictly from CSVs
    if os.path.exists(gemm_path):
        df_g = read_measured_csv(gemm_path)
        g_agg = df_g.groupby(['version', 'M'])[['flops_per_cycle']].mean().reset_index()

        sub_v0 = g_agg[g_agg['version'] == 'v0'].sort_values('M')
        sub_v1 = g_agg[g_agg['version'] == 'v1'].sort_values('M')

        # Plot v0 as hollow blue circles
        if not sub_v0.empty:
            i_gemm_v0 = sub_v0['M'].values / 2.0
            gflops_gemm_v0 = sub_v0['flops_per_cycle'].values * freq_ghz
            ax.scatter(i_gemm_v0, gflops_gemm_v0, facecolors='none', edgecolors='#1f77b4',
                       marker='o', s=85, linewidth=1.8, label='GEMM v0 Serial (M=1,2,4,8)', zorder=6)

        # Plot v1 as solid orange squares (at exact same I, naturally overlapping)
        if not sub_v1.empty:
            i_gemm_v1 = sub_v1['M'].values / 2.0
            gflops_gemm_v1 = sub_v1['flops_per_cycle'].values * freq_ghz
            ax.scatter(i_gemm_v1, gflops_gemm_v1, color='#ff7f0e', marker='s', s=45,
                       label='GEMM v1 Overlap (M=1,2,4,8)', zorder=7)

        for _, row in sub_v1.iterrows():
            m_val = int(row['M'])
            x_pt = m_val / 2.0
            y_pt = row['flops_per_cycle'] * freq_ghz
            ax.text(x_pt * 1.05, y_pt * 0.72, f"M={m_val}", fontsize=8.0, color='#333333', fontweight='bold')

    # Empirical Multicore 4-Core GEMM Point
    if os.path.exists(multicore_path):
        df_mc = read_measured_csv(multicore_path)
        sub_mc = df_mc[(df_mc['M'] == 8) & (df_mc['num_threads'] == 4) & (df_mc['exec_mode'] == 'N')]
        if not sub_mc.empty:
            mc_fpc = sub_mc['flops_per_cycle'].mean()  # 0.7555 FLOPs/cycle
            mc_gflops = mc_fpc * freq_ghz              # 1.2088 GFLOPS @ 1.6 GHz
            x_mc = 8.0 / 2.0                          # 4.0 FLOP/byte
            ax.scatter([x_mc], [mc_gflops], color='#6a3d9a', marker='*', s=160,
                       label=f'GEMM 4-Core M=8 ({mc_fpc:.4f} FLOPs/cyc, 2.72× Speedup)', zorder=8)
            ax.annotate('4-Core M=8 (2.72× 硬件加速)',
                        xy=(x_mc, mc_gflops), xytext=(x_mc * 1.15, mc_gflops * 1.35),
                        arrowprops=dict(facecolor='#6a3d9a', arrowstyle='->', lw=1.2),
                        fontsize=8.5, fontweight='bold', color='#6a3d9a')

    if os.path.exists(attn_path):
        df_a = read_measured_csv(attn_path)
        a_agg = df_a.groupby(['version', 'L'])[['flops', 'total_bytes', 'device_cycles']].mean().reset_index()

        sub_v0 = a_agg[a_agg['version'] == 'v0'].sort_values('L')
        if not sub_v0.empty:
            total_algo_bytes_v0 = 2 * sub_v0['L'].values * 64 * 4 + 512
            i_attn_v0 = sub_v0['flops'].values / total_algo_bytes_v0
            gflops_attn_v0 = (sub_v0['flops'].values / sub_v0['device_cycles'].values) * freq_ghz
            ax.scatter(i_attn_v0, gflops_attn_v0, facecolors='none', edgecolors='#d62728', marker='^', s=70, linewidth=1.8, label='Attn v0 DDR Stream (L=64..1024, I≈0.5)', zorder=6)

        sub_v1 = a_agg[a_agg['version'] == 'v1'].sort_values('L')
        if not sub_v1.empty:
            total_algo_bytes_v1 = 2 * sub_v1['L'].values * 64 * 4 + 512
            i_attn_v1 = sub_v1['flops'].values / total_algo_bytes_v1
            gflops_attn_v1 = (sub_v1['flops'].values / sub_v1['device_cycles'].values) * freq_ghz
            ax.scatter(i_attn_v1, gflops_attn_v1, color='#2ca02c', marker='D', s=45, label='Attn v1 DM Resident (L=64..1024, I≈0.5, ~9% speedup)', zorder=7)

        # Attention annotation
        ax.annotate("Attention v1 vs v0 (~9% SRAM 收益)\n算法算强 I ≈ 0.5 FLOP/B",
                    xy=(0.50, 0.171 * (freq_ghz / 1.425)), xytext=(0.06, 0.45),
                    arrowprops=dict(facecolor='#2ca02c', arrowstyle='->', lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.35", fc="#f0fff0", ec="#2ca02c", lw=1),
                    fontsize=8.5, fontweight='bold', color='#1b661b')

    # Empirical Vendor mablaslt Points
    if os.path.exists(vendor_gemm_path):
        df_vg = read_measured_csv(vendor_gemm_path)
        vg_agg = df_vg.groupby(['dtype', 'M'])[['flops_per_cycle']].mean().reset_index()

        sub_vg_fp32 = vg_agg[vg_agg['dtype'] == 'fp32'].sort_values('M')
        if not sub_vg_fp32.empty:
            i_vg_fp32 = sub_vg_fp32['M'].values / 2.0
            gflops_vg_fp32 = sub_vg_fp32['flops_per_cycle'].values * freq_ghz
            ax.scatter(i_vg_fp32, gflops_vg_fp32, color='#02818a', marker='P', s=80,
                       label='Vendor mablaslt FP32 (M=1..64)', zorder=8)
            pt_m8_fp32 = sub_vg_fp32[sub_vg_fp32['M'] == 8]
            if not pt_m8_fp32.empty:
                x_v = 4.0
                y_v = pt_m8_fp32['flops_per_cycle'].values[0] * freq_ghz
                ax.annotate(f"Vendor FP32 M=8\n({pt_m8_fp32['flops_per_cycle'].values[0]:.2f} F/cyc, 33.1× vs C)",
                            xy=(x_v, y_v), xytext=(x_v * 1.35, y_v * 0.75),
                            arrowprops=dict(facecolor='#02818a', arrowstyle='->', lw=1.1),
                            fontsize=8.0, fontweight='bold', color='#02818a')
            pt_m64_fp32 = sub_vg_fp32[sub_vg_fp32['M'] == 64]
            if not pt_m64_fp32.empty:
                x_v64 = 32.0
                y_v64 = pt_m64_fp32['flops_per_cycle'].values[0] * freq_ghz
                ax.annotate(f"Vendor FP32 M=64\n({pt_m64_fp32['flops_per_cycle'].values[0]:.1f} F/cyc, 30.2% peak)",
                            xy=(x_v64, y_v64), xytext=(x_v64 * 0.45, y_v64 * 1.35),
                            arrowprops=dict(facecolor='#02818a', arrowstyle='->', lw=1.1),
                            fontsize=8.0, fontweight='bold', color='#02818a')

        sub_vg_fp16 = vg_agg[vg_agg['dtype'] == 'fp16'].sort_values('M')
        if not sub_vg_fp16.empty:
            i_vg_fp16 = sub_vg_fp16['M'].values / 1.0
            gflops_vg_fp16 = sub_vg_fp16['flops_per_cycle'].values * freq_ghz
            ax.scatter(i_vg_fp16, gflops_vg_fp16, color='#e7298a', marker='X', s=85,
                       label='Vendor mablaslt FP16 (M=1..64)', zorder=8)
            pt_m8_fp16 = sub_vg_fp16[sub_vg_fp16['M'] == 8]
            if not pt_m8_fp16.empty:
                x_v16 = 8.0
                y_v16 = pt_m8_fp16['flops_per_cycle'].values[0] * freq_ghz
                ax.annotate(f"Vendor FP16 M=8\n({pt_m8_fp16['flops_per_cycle'].values[0]:.2f} F/cyc, 0.99% peak)",
                            xy=(x_v16, y_v16), xytext=(x_v16 * 1.25, y_v16 * 0.85),
                            arrowprops=dict(facecolor='#e7298a', arrowstyle='->', lw=1.1),
                            fontsize=8.0, fontweight='bold', color='#e7298a')
            pt_m64_fp16 = sub_vg_fp16[sub_vg_fp16['M'] == 64]
            if not pt_m64_fp16.empty:
                x_v64_16 = 64.0
                y_v64_16 = pt_m64_fp16['flops_per_cycle'].values[0] * freq_ghz
                ax.annotate(f"Vendor FP16 M=64\n({pt_m64_fp16['flops_per_cycle'].values[0]:.1f} F/cyc, 7.96% peak)",
                            xy=(x_v64_16, y_v64_16), xytext=(x_v64_16 * 0.45, y_v64_16 * 1.35),
                            arrowprops=dict(facecolor='#e7298a', arrowstyle='->', lw=1.1),
                            fontsize=8.0, fontweight='bold', color='#e7298a')

    # 3. 878x Three-layer gap vertical arrow and decomposition at x = 4.0 (GEMM M=8)
    x_gap = 4.0
    y_gemm8 = 0.2915 * freq_ghz    # 0.4664 GFLOPS (Empirical GEMM M=8 @ 1.6 GHz)
    y_cmodel = 0.432 * freq_ghz    # 0.6912 GFLOPS (Compiled C vector loop)
    y_asm = 32.0 * freq_ghz        # 51.200 GFLOPS (Handwritten VLIW Asm)
    y_peak = 256.0 * freq_ghz      # 409.60 GFLOPS (Single APE FP32 Peak Roof)

    # Draw vertical dashed guide line with ticks
    ax.plot([x_gap, x_gap], [y_gemm8, y_peak], color='#990000', linestyle='--', linewidth=1.5, zorder=8)
    for y_pt in [y_gemm8, y_cmodel, y_asm, y_peak]:
        ax.plot([x_gap * 0.94, x_gap * 1.06], [y_pt, y_pt], color='#990000', linewidth=1.5, zorder=8)

    # Arrowhead at top
    ax.annotate('', xy=(x_gap, y_peak), xytext=(x_gap, y_peak * 0.85),
                arrowprops=dict(arrowstyle="->", color="#990000", lw=1.8), zorder=9)

    # 3 segment annotations - shortened and tight to ruler right side
    ax.text(x_gap * 1.07, 140.0, "① 8× 发射槽",
            fontsize=8.5, fontweight='bold', color='#990000', va='center')

    y_t2 = np.sqrt(y_cmodel * y_asm)  # 5.95 GFLOPS
    ax.text(x_gap * 1.07, y_t2, "② 74× 编译器",
            fontsize=8.5, fontweight='bold', color='#990000', va='center')

    y_t3 = np.sqrt(y_gemm8 * y_cmodel)  # 0.568 GFLOPS
    ax.text(x_gap * 1.07, y_t3, "③ 1.48× 控制开销",
            fontsize=8.5, fontweight='bold', color='#990000', va='center')

    # Overall 878x header box in left open area (x≈0.18, y≈35), with curved thin leader line to ruler top
    ax.annotate("【878× 算力差距因数分解】\n8.00 × 74.00 × 1.481 = 878×\n(注: FLOPs/cyc 标尺与频率无关)",
                xy=(x_gap, y_peak), xytext=(0.18, 35.0),
                arrowprops=dict(arrowstyle="->", color="#990000", lw=1.0, linestyle="--",
                                connectionstyle="arc3,rad=-0.12"),
                fontsize=8.0, fontweight='bold', color='#990000', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.35", fc="#fff2f2", ec="#990000", lw=1.2),
                zorder=10)

    # Note text outside below legend
    note_text = (
        "【多精度屋顶与平衡拐点】\n"
        "- INT8:  840 TFLOPS, R = 256.3 FLOP/B\n"
        "- FP16:  420 TFLOPS, R = 128.2 FLOP/B\n"
        "- FP32: 52.5 TFLOPS, R =  16.0 FLOP/B\n"
        "- 均摊带宽: 25.6 GB/s (整Die 3276.8 GB/s)\n\n"
        "【多核扩展实测 (M=8)】\n"
        "4核达 0.7555 FLOPs/cyc (2.72× 加速比)\n\n"
        "【厂商 mablaslt 对照 (M=8)】\n"
        "FP32:  9.65 FLOPs/cyc (33.1× vs C)\n"
        "FP16: 20.22 FLOPs/cyc (0.99% 峰值)\n"
        "M≤8 硬件周期恒定 (分块 Tile 流水)\n\n"
        "【878× 差距分解 (频率无关)】\n"
        "① 8× 发射槽: 微码仅发射 1 路向量加法\n"
        "② 74× 编译器: 缺乏软件流水与栈溢出\n"
        "③ 1.48× 控制开销: 循环控制与 DMA 调度"
    )
    ax.text(1.02, 0.22, note_text, transform=ax.transAxes,
            fontsize=7.8, color='#333333', va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#cccccc", lw=0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(i_min, i_max)
    ax.set_ylim(0.05, 25000)

    # Clean log format for both axes
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(clean_log_format))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(clean_log_format))

    ax.set_xlabel('Arithmetic Intensity I (FLOP / Byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attainable Compute Throughput (GFLOPS, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Target LPU Architecture Roofline Model: Nominal Multi-Precision Roofs vs. CModel Kernels',
                 fontsize=13, fontweight='bold', pad=14)

    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Place legend outside on the right top
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=True, framealpha=0.92, fontsize=8.5)

    out_file = os.path.join(FIGURES_DIR, "roofline.png")
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Generated {out_file}")

def main():
    print("Generating benchmark figures...")
    plot_k1_gemm()
    plot_k2_attn()
    plot_k2_breakdown()
    plot_k3_speedup()
    plot_roofline()
    print("Done!")

if __name__ == "__main__":
    main()
