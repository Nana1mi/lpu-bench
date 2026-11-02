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

def plot_k1_gemm():
    csv_path = os.path.join(RESULTS_DIR, "gemm.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} does not exist, skipping K1 plot")
        return

    df = pd.read_csv(csv_path)
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

    df = pd.read_csv(csv_path)
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

    df = pd.read_csv(csv_path)
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
        df_g = pd.read_csv(gemm_path)
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
        df_a = pd.read_csv(attn_path)
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

def main():
    print("Generating benchmark figures...")
    plot_k1_gemm()
    plot_k2_attn()
    plot_k2_breakdown()
    plot_k3_speedup()
    print("Done!")

if __name__ == "__main__":
    main()
