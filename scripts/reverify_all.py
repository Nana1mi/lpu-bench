#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Baselines to verify bit-exact parity
ATTN_BASELINES = {
    ("v0", 256): 589837,
    ("v2", 256): 593914,
    ("v0", 512): 1176205,
    ("v2", 512): 1184458,
    ("v0", 1024): 2348945,
    ("v1", 1024): 2167110,
    ("v2", 1024): 2365550,
}

GEMM_BASELINES = {
    ("v0", 1): 470550,
    ("v1", 1): 472466,
    ("v0", 2): 917183,
    ("v1", 2): 919237,
    ("v0", 4): 1810449,
    ("v1", 4): 1812443,
    ("v0", 8): 3596981,
    ("v1", 8): 3598855,
}

def run_cmd(cmd, desc):
    print(f"\n=======================================================")
    print(f"  [START] {desc}")
    print(f"  Command: {cmd}")
    print(f"=======================================================", flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"[ERROR] Command failed with return code {ret.returncode} ({elapsed:.1f}s)", file=sys.stderr)
        sys.exit(ret.returncode)
    print(f"  [DONE] {desc} finished in {elapsed:.1f}s\n", flush=True)

def verify_results():
    print("\n=======================================================")
    print("         VERIFICATION & COMPARISON REPORT              ")
    print("=======================================================")

    # 1. Decode Attn Verification
    attn_csv = os.path.join(RESULTS_DIR, "decode_attn.csv")
    df_a = pd.read_csv(attn_csv, keep_default_na=False)
    # Filter for newly measured rows (non-empty build_id)
    measured_a = df_a[df_a['build_id'] != '']
    print(f"\n[1] Decode Attention ({len(measured_a)} total measured rows):")
    attn_mismatches = 0
    for (ver, L), expected in ATTN_BASELINES.items():
        sub = measured_a[(measured_a['version'] == ver) & (measured_a['L'] == L)]
        if sub.empty:
            print(f"  [MISSING] {ver} L={L}")
            attn_mismatches += 1
            continue
        actual_cycles = sub['device_cycles'].tolist()
        unique_cycles = set(actual_cycles)
        if len(unique_cycles) == 1 and actual_cycles[0] == expected:
            print(f"  [MATCH] {ver} L={L:4d}: {actual_cycles[0]} cycles (3/3 repeats identical, matches baseline {expected})")
        else:
            print(f"  [MISMATCH/DIFF] {ver} L={L:4d}: expected {expected}, got {actual_cycles}")
            attn_mismatches += 1

    # 2. GEMM Verification
    gemm_csv = os.path.join(RESULTS_DIR, "gemm.csv")
    df_g = pd.read_csv(gemm_csv, keep_default_na=False)
    measured_g = df_g[df_g['build_id'] != '']
    print(f"\n[2] GEMM ({len(measured_g)} total measured rows):")
    gemm_mismatches = 0
    for (ver, M), expected in GEMM_BASELINES.items():
        sub = measured_g[(measured_g['version'] == ver) & (measured_g['M'] == M)]
        if sub.empty:
            print(f"  [MISSING] {ver} M={M}")
            gemm_mismatches += 1
            continue
        actual_cycles = sub['device_cycles'].tolist()
        unique_cycles = set(actual_cycles)
        if len(unique_cycles) == 1 and actual_cycles[0] == expected:
            print(f"  [MATCH] {ver} M={M:2d}: {actual_cycles[0]} cycles (3/3 repeats identical, matches baseline {expected})")
        else:
            print(f"  [MISMATCH/DIFF] {ver} M={M:2d}: expected {expected}, got {actual_cycles}")
            gemm_mismatches += 1

    print("\n-------------------------------------------------------")
    if attn_mismatches == 0 and gemm_mismatches == 0:
        print(">>> ALL BASELINES REPRODUCED BIT-FOR-BIT WITH ZERO VARIANCE! <<<")
    else:
        print(f">>> ATTENTION MISMATCHES: {attn_mismatches}, GEMM MISMATCHES: {gemm_mismatches} <<<")
    print("-------------------------------------------------------\n")

def main():
    total_start = time.time()

    # Step 1: Decode Attn v0, v2 @ L=256, 512
    run_cmd("python3 scripts/run_k2_attn.py --filter-version v0,v2 --filter-l 256,512",
            "Decode Attention (v0, v2 @ L=256, 512)")

    # Step 2: Decode Attn v0, v1, v2 @ L=1024
    run_cmd("python3 scripts/run_k2_attn.py --filter-l 1024",
            "Decode Attention (v0, v1, v2 @ L=1024)")

    # Step 3: GEMM v0, v1 @ M=1, 2, 4, 8
    run_cmd("python3 scripts/run_k1_gemm.py",
            "GEMM (v0, v1 @ M=1, 2, 4, 8)")

    # Step 4: Verification against baseline numbers
    verify_results()

    # Step 5: Regenerate Ablation and Plots
    run_cmd("python3 scripts/run_k3_ablation.py", "Regenerate Ablation Results")
    run_cmd("python3 scripts/plot.py", "Regenerate Plots")

    total_time = time.time() - total_start
    print(f"\n[FINISHED] Entire reverification workflow completed in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()
