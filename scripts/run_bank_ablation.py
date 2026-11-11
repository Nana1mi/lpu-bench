#!/usr/bin/env python3
"""
run_bank_ablation.py - Benchmark suite for Bank Layout and Region Size Ablation (T2).
Measures:
1. GEMM default vs explicit (__fix1~3__) vs intentional conflict (__fix1__ on W and X) for M in [1, 4, 8]
2. Attention compact (32KB region) vs large (256KB region) on L=128
Outputs to results/bank_ablation.csv.
"""

import os
import sys
import subprocess
import csv
import datetime
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUEST_RUN = os.path.join(PROJECT_ROOT, "scripts", "guest_run.sh")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "bank_ablation.csv")
BINARY_PATH = os.path.join(PROJECT_ROOT, "kernels", "bank_ablation", "bank_ablation.out")

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"

def get_binary_md5(binary_path):
    try:
        with open(binary_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return "unknown"

def get_timestamp():
    return datetime.datetime.now().isoformat()

CONFIGS = [
    # GEMM Bank Mapping Ablation (M in 1, 4, 8)
    ("gemm", "gemm_default", "default", 1, 256, 256, 128, 64, 3),
    ("gemm", "gemm_default", "default", 4, 256, 256, 128, 64, 3),
    ("gemm", "gemm_default", "default", 8, 256, 256, 128, 64, 3),

    ("gemm", "gemm_explicit", "explicit_fix123", 1, 256, 256, 128, 64, 3),
    ("gemm", "gemm_explicit", "explicit_fix123", 4, 256, 256, 128, 64, 3),
    ("gemm", "gemm_explicit", "explicit_fix123", 8, 256, 256, 128, 64, 3),

    ("gemm", "gemm_conflict", "conflict_fix112", 1, 256, 256, 128, 64, 3),
    ("gemm", "gemm_conflict", "conflict_fix112", 4, 256, 256, 128, 64, 3),
    ("gemm", "gemm_conflict", "conflict_fix112", 8, 256, 256, 128, 64, 3),

    # Attention Region Size Ablation (L=128, d=64)
    ("attention", "attn_compact", "compact_32KB", 1, 0, 0, 128, 64, 3),
    ("attention", "attn_large", "large_256KB", 1, 0, 0, 128, 64, 3),
]

def run_test(test_name, M, K, N, L, repeats):
    guest_cmd = (
        f"cd /home/emu/lpu-workspace/lpu-bench/kernels/bank_ablation && "
        f"MAPS_DUMP_GPU_DEBUG_INFO_ENABLE=false MALOG_LEVEL=error "
        f"./bank_ablation.out --test {test_name} --M {M} --K {K} --N {N} --L {L} --repeats {repeats}"
    )
    full_cmd = f"{GUEST_RUN} '{guest_cmd}'"
    print(f"\n[RUN] Executing: test={test_name}, M={M}, L={L}, repeats={repeats} ...", flush=True)

    proc = subprocess.run(["bash", "-c", full_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] Return code: {proc.returncode}\nStderr: {proc.stderr}", file=sys.stderr)
        return []

    lines = proc.stdout.splitlines()
    csv_lines = []
    in_csv = False
    for line in lines:
        if "--- CSV DATA START ---" in line:
            in_csv = True
            continue
        if "--- CSV DATA END ---" in line:
            in_csv = False
            continue
        if in_csv and line.strip():
            sline = line.strip()
            if sline.startswith("test,") or sline.startswith("gemm_") or sline.startswith("attn_"):
                csv_lines.append(sline)

    return csv_lines

def main():
    print("==========================================================")
    print("   LPU-Bench: Bank Layout & Region Size Ablation Suite    ")
    print("==========================================================")

    build_id = get_binary_md5(BINARY_PATH)
    commit_hash = get_git_commit()
    print(f"Binary Path: {BINARY_PATH}")
    print(f"Build ID (MD5): {build_id}")
    print(f"Git Commit: {commit_hash}")

    write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    header = None

    for kernel_type, test_name, config_label, M, K, N, L, d, repeats in CONFIGS:
        output_lines = run_test(test_name, M, K, N, L, repeats)
        if not output_lines:
            print(f"[ERROR] No output for test={test_name} M={M}", file=sys.stderr)
            continue

        cur_header = "kernel,experiment,config,M,K,N,repeat,sim_ms,sim_cycles,device_cycles,wall_ms,correct,max_diff,commit_hash,build_id,timestamp"
        if header is None:
            header = cur_header
            if write_header:
                with open(RESULTS_CSV, "w", newline="") as f:
                    f.write(header + "\n")
                write_header = False

        data_rows = output_lines[1:]
        timestamp = get_timestamp()
        with open(RESULTS_CSV, "a", newline="") as f:
            for row in data_rows:
                parts = row.split(",")
                # row is: test,config,M,K,N,repeat,sim_ms,sim_cycles,device_cycles,wall_ms,correct,max_diff
                out_row = f"{kernel_type},{parts[0]},{parts[1]},{','.join(parts[2:])},{commit_hash},{build_id},{timestamp}"
                f.write(out_row + "\n")
                print(f"  -> Recorded: {out_row}")

    print("\n[COMPLETE] Bank ablation benchmark finished! Results written to:", RESULTS_CSV)

if __name__ == "__main__":
    main()
