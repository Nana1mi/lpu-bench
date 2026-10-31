#!/usr/bin/env python3
import os
import sys
import subprocess
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUEST_RUN = os.path.join(PROJECT_ROOT, "scripts", "guest_run.sh")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "gemm.csv")

os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

CONFIGS = [
    # (version, M, K, N, repeats)
    # M=1, 2, 4, 8 evaluated across 3 repeats (Zero variance in gem5 cycle simulation)
    # Larger M (16, 32) take >30 mins per run in gem5 CModel and are pruned per thesis rules
    ("v0", 1, 256, 256, 3),
    ("v1", 1, 256, 256, 3),
    ("v0", 2, 256, 256, 3),
    ("v1", 2, 256, 256, 3),
    ("v0", 4, 256, 256, 3),
    ("v1", 4, 256, 256, 3),
    ("v0", 8, 256, 256, 3),
    ("v1", 8, 256, 256, 3),
]

def run_config(version, M, K, N, repeats):
    guest_cmd = f"cd /home/emu/lpu-workspace/lpu-bench/kernels/gemm && MAPS_DUMP_GPU_DEBUG_INFO_ENABLE=false MALOG_LEVEL=error ./gemm.out {version} {M} {K} {N} {repeats}"
    full_cmd = f"{GUEST_RUN} '{guest_cmd}'"
    print(f"\n[RUN] Executing: {version}, M={M}, K={K}, N={N}, repeats={repeats} ...", flush=True)
    
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
            if sline.startswith("kernel,") or sline.startswith("gemm,"):
                csv_lines.append(sline)

    return csv_lines

def main():
    print("==========================================================")
    print("      LPU-Bench: K1 GEMM Scaling Benchmark Suite          ")
    print("==========================================================")
    
    write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    header = None

    for version, M, K, N, repeats in CONFIGS:
        output_lines = run_config(version, M, K, N, repeats)
        if not output_lines:
            print(f"[ERROR] No output for {version} M={M}", file=sys.stderr)
            continue
            
        cur_header = output_lines[0]
        if header is None:
            header = cur_header
            if write_header:
                with open(RESULTS_CSV, "w", newline="") as f:
                    f.write(header + "\n")
                write_header = False

        data_rows = output_lines[1:]
        with open(RESULTS_CSV, "a", newline="") as f:
            for row in data_rows:
                f.write(row + "\n")
                print(f"  -> Recorded: {row}")

    print("\n[COMPLETE] K1 Benchmark finished! Results written to:", RESULTS_CSV)

if __name__ == "__main__":
    main()
