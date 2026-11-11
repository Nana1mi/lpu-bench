#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import datetime
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUEST_RUN = os.path.join(PROJECT_ROOT, "scripts", "guest_run.sh")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "gemm_multicore.csv")
BINARY_PATH = os.path.join(PROJECT_ROOT, "kernels", "gemm_multicore", "gemm_multicore.out")

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

os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

# Test matrix:
# N mode: 1, 2, 4 threads x M in [1, 2, 4, 8]
# C mode: 4 threads x M in [1, 2, 4, 8]
CONFIGS = [
    # (mode, num_threads, M, K, N, repeats)
    # N mode
    ("N", 1, 1, 256, 256, 3),
    ("N", 1, 2, 256, 256, 3),
    ("N", 1, 4, 256, 256, 3),
    ("N", 1, 8, 256, 256, 3),

    ("N", 2, 1, 256, 256, 3),
    ("N", 2, 2, 256, 256, 3),
    ("N", 2, 4, 256, 256, 3),
    ("N", 2, 8, 256, 256, 3),

    ("N", 4, 1, 256, 256, 3),
    ("N", 4, 2, 256, 256, 3),
    ("N", 4, 4, 256, 256, 3),
    ("N", 4, 8, 256, 256, 3),

    # C mode (exclusive APC 4-thread)
    ("C", 4, 1, 256, 256, 3),
    ("C", 4, 2, 256, 256, 3),
    ("C", 4, 4, 256, 256, 3),
    ("C", 4, 8, 256, 256, 3),
]

def run_config(mode, num_threads, M, K, N, repeats):
    guest_cmd = (
        f"cd /home/emu/lpu-workspace/lpu-bench/kernels/gemm_multicore && "
        f"MAPS_DUMP_GPU_DEBUG_INFO_ENABLE=false MALOG_LEVEL=error "
        f"./gemm_multicore.out --mode {mode} --threads {num_threads} --M {M} --K {K} --N {N} --repeats {repeats}"
    )
    full_cmd = f"{GUEST_RUN} '{guest_cmd}'"
    print(f"\n[RUN] Executing: mode={mode}, threads={num_threads}, M={M}, K={K}, N={N}, repeats={repeats} ...", flush=True)

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
    print("   LPU-Bench: K1 GEMM Multi-Core Scaling Benchmark Suite  ")
    print("==========================================================")

    build_id = get_binary_md5(BINARY_PATH)
    commit_hash = get_git_commit()
    print(f"Binary Path: {BINARY_PATH}")
    print(f"Build ID (MD5): {build_id}")
    print(f"Git Commit: {commit_hash}")

    write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    header = None

    for mode, num_threads, M, K, N, repeats in CONFIGS:
        output_lines = run_config(mode, num_threads, M, K, N, repeats)
        if not output_lines:
            print(f"[ERROR] No output for mode={mode} threads={num_threads} M={M}", file=sys.stderr)
            continue

        cur_header = output_lines[0] + ",commit_hash,build_id,timestamp"
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
                out_row = f"{row},{commit_hash},{build_id},{timestamp}"
                f.write(out_row + "\n")
                print(f"  -> Recorded: {out_row}")

    print("\n[COMPLETE] Multi-core GEMM benchmark finished! Results written to:", RESULTS_CSV)

if __name__ == "__main__":
    main()
