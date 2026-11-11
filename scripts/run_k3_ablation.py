#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ABLATION_CSV = os.path.join(RESULTS_DIR, "ablation.csv")

def main():
    print("==========================================================")
    print("       LPU-Bench: K3 Pipelined Overlap & Ablation Suite   ")
    print("==========================================================")

    gemm_csv = os.path.join(RESULTS_DIR, "gemm.csv")
    attn_csv = os.path.join(RESULTS_DIR, "decode_attn.csv")

    ablation_records = []

    # 1. Process GEMM Overlap Ablation
    if os.path.exists(gemm_csv):
        df_g = pd.read_csv(gemm_csv)
        if 'build_id' in df_g.columns and df_g['build_id'].fillna('').astype(str).str.strip().ne('').any():
            df_g = df_g[df_g['build_id'].fillna('').astype(str).str.strip() != '']
        # Filter valid numeric rows
        df_g = df_g[df_g['kernel'] == 'gemm']
        df_g['device_cycles'] = pd.to_numeric(df_g['device_cycles'], errors='coerce')
        df_g['M'] = pd.to_numeric(df_g['M'], errors='coerce')
        g_agg = df_g.groupby(['M', 'version'])['device_cycles'].mean().unstack(level=1)

        print("\n[GEMM Pipelined Overlap Ablation]")
        for M in g_agg.index:
            if 'v0' in g_agg.columns and 'v1' in g_agg.columns:
                c_v0 = g_agg.loc[M, 'v0']
                c_v1 = g_agg.loc[M, 'v1']
                speedup = c_v0 / c_v1 if c_v1 > 0 else 1.0
                diff = c_v0 - c_v1
                ablation_records.append({
                    'category': 'GEMM_Overlap',
                    'param_name': 'M',
                    'param_value': M,
                    'baseline_version': 'v0_serial',
                    'baseline_cycles': int(c_v0),
                    'target_version': 'v1_overlap',
                    'target_cycles': int(c_v1),
                    'speedup': round(speedup, 4),
                    'cycles_saved': int(diff),
                    'notes': f'M={M}, K=256, N=256 double buffering'
                })
                print(f"  M={M:2d}: v0={int(c_v0):d} cyc, v1={int(c_v1):d} cyc, Speedup={speedup:.4f}x (saved {int(diff):d} cyc)")

    # 2. Process Decode Attention Overlap & SRAM Locality Ablation
    if os.path.exists(attn_csv):
        df_a = pd.read_csv(attn_csv)
        if 'build_id' in df_a.columns and df_a['build_id'].fillna('').astype(str).str.strip().ne('').any():
            df_a = df_a[df_a['build_id'].fillna('').astype(str).str.strip() != '']
        df_a = df_a[df_a['kernel'] == 'decode_attn']
        df_a['device_cycles'] = pd.to_numeric(df_a['device_cycles'], errors='coerce')
        df_a['L'] = pd.to_numeric(df_a['L'], errors='coerce')
        a_agg = df_a.groupby(['L', 'version'])['device_cycles'].mean().unstack(level=1)

        print("\n[Decode Attention Overlap & Locality Ablation]")
        for L in a_agg.index:
            c_v0 = a_agg.loc[L, 'v0'] if 'v0' in a_agg.columns else None
            c_v1 = a_agg.loc[L, 'v1'] if 'v1' in a_agg.columns else None
            c_v2 = a_agg.loc[L, 'v2'] if 'v2' in a_agg.columns else None

            # Overlap ablation (v0 vs v2)
            if c_v0 is not None and c_v2 is not None and not np.isnan(c_v2) and not np.isnan(c_v0):
                speedup_ov = c_v0 / c_v2 if c_v2 > 0 else 1.0
                diff_ov = c_v0 - c_v2
                ablation_records.append({
                    'category': 'Attn_DDR_Overlap',
                    'param_name': 'L',
                    'param_value': L,
                    'baseline_version': 'v0_ddr',
                    'baseline_cycles': int(c_v0),
                    'target_version': 'v2_overlap',
                    'target_cycles': int(c_v2),
                    'speedup': round(speedup_ov, 4),
                    'cycles_saved': int(diff_ov),
                    'notes': f'L={L}, d=64 DDR double-buffered streaming'
                })
                print(f"  L={L:3d} (v0 vs v2): v0={int(c_v0):d} cyc, v2={int(c_v2):d} cyc, Speedup={speedup_ov:.4f}x")

            # Locality ablation (v0 vs v1 resident SRAM)
            if c_v0 is not None and c_v1 is not None and not np.isnan(c_v1) and not np.isnan(c_v0):
                speedup_res = c_v0 / c_v1 if c_v1 > 0 else 1.0
                dma_overhead = c_v0 - c_v1
                dma_pct = (dma_overhead / c_v0) * 100.0 if c_v0 > 0 else 0.0
                ablation_records.append({
                    'category': 'Attn_SRAM_Locality',
                    'param_name': 'L',
                    'param_value': L,
                    'baseline_version': 'v0_ddr',
                    'baseline_cycles': int(c_v0),
                    'target_version': 'v1_resident',
                    'target_cycles': int(c_v1),
                    'speedup': round(speedup_res, 4),
                    'cycles_saved': int(dma_overhead),
                    'notes': f'L={L}, d=64 resident on-chip DM SRAM ({dma_pct:.1f}% DMA overhead)'
                })
                print(f"  L={L:3d} (v0 vs v1): v0={int(c_v0):d} cyc, v1={int(c_v1):d} cyc, Speedup={speedup_res:.4f}x (DMA overhead: {dma_pct:.1f}%)")

    if ablation_records:
        df_out = pd.DataFrame(ablation_records)
        df_out.to_csv(ABLATION_CSV, index=False)
        print(f"\n[COMPLETE] Ablation data written to: {ABLATION_CSV}")
    else:
        print("\n[WARN] No ablation records generated (check input CSVs)")

if __name__ == "__main__":
    main()
