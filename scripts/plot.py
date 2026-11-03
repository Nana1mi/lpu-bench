#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
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

    plt.title('K1: Aurora GEMM Compute Throughput vs. Batch/Token Scale M', fontsize=14, fontweight='bold', pad=15)
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

def plot_roofline():
    gemm_path = os.path.join(RESULTS_DIR, "gemm.csv")
    attn_path = os.path.join(RESULTS_DIR, "decode_attn.csv")

    fig, ax = plt.subplots(figsize=(10.5, 7.0), dpi=300)

    # 1. Hardware Specifications (Vendor derived)
    peak_bw_chip = 512.0          # GB/s (Whole chip GDDR6)
    peak_bw_ape_amortized = 512.0 / 16.0  # 32.0 GB/s (Amortized per APE)
    peak_flops_chip = 5836.8      # GFLOPS (5.837 TFLOPS)
    peak_flops_ape = 364.8        # GFLOPS (Single APE)
    freq_ghz = 1.425

    # Knee points (R = FLOPs / Byte)
    r_chip = peak_flops_chip / peak_bw_chip                  # 11.40 FLOP/byte
    r_ape_amortized = peak_flops_ape / peak_bw_ape_amortized  # 11.40 FLOP/byte (Scale-invariant)
    r_ape_exclusive = peak_flops_ape / peak_bw_chip          # 0.7125 FLOP/byte (Optimistic upper bound)

    # X range for intensity
    i_min = 0.05
    i_max = 64.0
    i_vals = np.logspace(np.log10(i_min), np.log10(i_max), 500)

    # Roofline curves
    roof_chip = np.minimum(peak_bw_chip * i_vals, peak_flops_chip)
    roof_ape_amortized = np.minimum(peak_bw_ape_amortized * i_vals, peak_flops_ape)
    roof_ape_exclusive = np.minimum(peak_bw_chip * i_vals, peak_flops_ape)

    # Plot roofs
    ax.plot(i_vals, roof_chip, 'k-', linewidth=2.0, label='Whole Chip Roof (16 APE, 5.837 TFLOPS, 512 GB/s, R=11.40)')
    ax.plot(i_vals, roof_ape_amortized, 'b-', linewidth=1.8, label='Single APE Amortized Roof (32 GB/s, 364.8 GFLOPS, R=11.40)')
    ax.plot(i_vals, roof_ape_exclusive, 'b:', linewidth=1.5, label='Single APE Exclusive Roof (512 GB/s Optimistic Upper Bound, R=0.71)')

    # Bandwidth slope texts
    ax.text(0.08, 512 * 0.08 * 1.35, 'Chip Peak BW: 512 GB/s', rotation=38, fontsize=8.5, fontweight='bold', color='#333333')
    ax.text(0.16, 32 * 0.16 * 0.72, 'APE Amortized BW: 32 GB/s', rotation=38, fontsize=8.5, fontweight='bold', color='blue')

    # Knee points
    ax.scatter([r_chip], [peak_flops_chip], color='black', s=60, zorder=5)
    ax.annotate(f'Whole Chip Knee\nR={r_chip:.2f} FLOP/B\n(5.837 TFLOPS)',
                xy=(r_chip, peak_flops_chip), xytext=(r_chip * 1.35, peak_flops_chip * 0.55),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                fontsize=8.5, fontweight='bold')

    ax.scatter([r_ape_amortized], [peak_flops_ape], color='blue', s=60, zorder=5)
    ax.annotate(f'Single APE Amortized Knee\nR={r_ape_amortized:.2f} FLOP/B\n(364.8 GFLOPS, 32 GB/s)',
                xy=(r_ape_amortized, peak_flops_ape), xytext=(r_ape_amortized * 1.35, peak_flops_ape * 0.40),
                arrowprops=dict(facecolor='blue', arrowstyle='->', lw=1.2),
                fontsize=8.5, fontweight='bold', color='blue')

    ax.scatter([r_ape_exclusive], [peak_flops_ape], color='blue', marker='x', s=65, zorder=5)
    ax.annotate(f'Single APE Exclusive Knee (Upper Bound)\nR={r_ape_exclusive:.2f} FLOP/B (512 GB/s)',
                xy=(r_ape_exclusive, peak_flops_ape), xytext=(r_ape_exclusive * 0.35, peak_flops_ape * 1.35),
                arrowprops=dict(facecolor='blue', arrowstyle='->', lw=1.0),
                fontsize=8.0, fontweight='bold', color='#003399')

    # Vertical reference lines
    ax.axvline(0.5, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(0.5 * 1.08, 0.10, 'Decode Attention\nI ≈ 0.5 FLOP/B (Fixed)', color='purple', fontsize=8.5, fontweight='bold')

    ax.axvline(23.0 / 2.0, color='darkgreen', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text((23.0 / 2.0) * 1.05, 0.10, 'FFN M=23 Knee\nI ≈ 11.5 FLOP/B', color='darkgreen', fontsize=8.5, fontweight='bold')

    # 2. Empirical Measured Data Points strictly from CSVs
    if os.path.exists(gemm_path):
        df_g = read_measured_csv(gemm_path)
        g_agg = df_g.groupby(['version', 'M'])[['flops_per_cycle']].mean().reset_index()

        # GEMM v0: hollow blue circle
        sub_v0 = g_agg[g_agg['version'] == 'v0'].sort_values('M')
        if not sub_v0.empty:
            i_gemm_v0 = sub_v0['M'].values / 2.0  # I = M / 2
            gflops_gemm_v0 = sub_v0['flops_per_cycle'].values * freq_ghz
            ax.scatter(i_gemm_v0, gflops_gemm_v0, facecolors='none', edgecolors='#1f77b4', marker='o', s=75, linewidth=1.8, label='GEMM v0 Serial (M=1,2,4,8)', zorder=6)
            for _, row in sub_v0.iterrows():
                m_val = int(row['M'])
                x_pt = m_val / 2.0
                y_pt = row['flops_per_cycle'] * freq_ghz
                ax.text(x_pt * 0.88, y_pt * 1.18, f"M={m_val}(v0)", fontsize=7.5, color='#1f77b4', fontweight='bold')

        # GEMM v1: solid orange square
        sub_v1 = g_agg[g_agg['version'] == 'v1'].sort_values('M')
        if not sub_v1.empty:
            i_gemm_v1 = sub_v1['M'].values / 2.0  # I = M / 2
            gflops_gemm_v1 = sub_v1['flops_per_cycle'].values * freq_ghz
            ax.scatter(i_gemm_v1, gflops_gemm_v1, color='#ff7f0e', marker='s', s=45, label='GEMM v1 Overlap (M=1,2,4,8)', zorder=7)
            for _, row in sub_v1.iterrows():
                m_val = int(row['M'])
                x_pt = m_val / 2.0
                y_pt = row['flops_per_cycle'] * freq_ghz
                ax.text(x_pt * 1.05, y_pt * 0.82, f"M={m_val}(v1)", fontsize=7.5, color='#ff7f0e', fontweight='bold')

    if os.path.exists(attn_path):
        df_a = read_measured_csv(attn_path)
        a_agg = df_a.groupby(['version', 'L'])[['flops', 'total_bytes', 'device_cycles']].mean().reset_index()

        # For Decode Attention, both v0 and v1 use complete algorithmic memory traffic:
        # 2 * L * d * 4 + 512 bytes (with d=64, 512*L + 512 bytes)
        # This places both v0 and v1 at I ≈ 0.5 FLOP/byte
        sub_v0 = a_agg[a_agg['version'] == 'v0'].sort_values('L')
        if not sub_v0.empty:
            total_algo_bytes_v0 = 2 * sub_v0['L'].values * 64 * 4 + 512
            i_attn_v0 = sub_v0['flops'].values / total_algo_bytes_v0
            gflops_attn_v0 = (sub_v0['flops'].values / sub_v0['device_cycles'].values) * freq_ghz
            ax.scatter(i_attn_v0, gflops_attn_v0, facecolors='none', edgecolors='#d62728', marker='^', s=65, linewidth=1.8, label='Attn v0 DDR Stream (L=64..1024, I≈0.5)', zorder=6)

        sub_v1 = a_agg[a_agg['version'] == 'v1'].sort_values('L')
        if not sub_v1.empty:
            total_algo_bytes_v1 = 2 * sub_v1['L'].values * 64 * 4 + 512
            i_attn_v1 = sub_v1['flops'].values / total_algo_bytes_v1
            gflops_attn_v1 = (sub_v1['flops'].values / sub_v1['device_cycles'].values) * freq_ghz
            ax.scatter(i_attn_v1, gflops_attn_v1, color='#2ca02c', marker='D', s=45, label='Attn v1 DM Resident (L=64..1024, I≈0.5, ~9% speedup)', zorder=7)

        # Annotation highlighting the ~9% on-chip residency speedup at I≈0.5
        ax.annotate("Attention v1 vs v0 (~9% SRAM speedup)\nBoth at algorithmic I ≈ 0.5 FLOP/B",
                    xy=(0.50, 0.171), xytext=(0.09, 0.45),
                    arrowprops=dict(facecolor='#2ca02c', arrowstyle='->', lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.35", fc="#f0fff0", ec="#2ca02c", lw=1),
                    fontsize=8.5, fontweight='bold', color='#1b661b')

    # Empirical cluster annotation showing the ~1/878 gap
    ax.annotate("Empirical Measured Points (Single APE)\n0.16 - 0.42 GFLOPS (~1/878 of Single APE Peak)",
                xy=(1.0, 0.35), xytext=(0.07, 1.4),
                arrowprops=dict(facecolor='#d62728', arrowstyle='->', lw=1.2),
                bbox=dict(boxstyle="round,pad=0.4", fc="#fff2f2", ec="#d62728", lw=1),
                fontsize=9, fontweight='bold', color='#990000')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(i_min, i_max)
    ax.set_ylim(0.08, 15000)

    ax.set_xlabel('Arithmetic Intensity I (FLOP / Byte)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attainable Compute Throughput (GFLOPS, log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Aurora Architecture Roofline Model: Nominal Roofs vs. CModel Measured Kernels',
                 fontsize=13, fontweight='bold', pad=14)

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='lower right', frameon=True, framealpha=0.92, fontsize=8.2)

    out_file = os.path.join(FIGURES_DIR, "roofline.png")
    fig.tight_layout()
    fig.savefig(out_file)
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
