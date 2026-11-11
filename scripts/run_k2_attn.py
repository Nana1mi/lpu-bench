#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import datetime
import argparse
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUEST_RUN = os.path.join(PROJECT_ROOT, "scripts", "guest_run.sh")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "decode_attn.csv")
BINARY_PATH = os.path.join(PROJECT_ROOT, "kernels", "decode_attn", "decode_attn.out")

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

ALL_CONFIGS = [
    # (version, L, d, repeats)
    ("v0", 64, 64, 3),
    ("v1", 64, 64, 3),
    ("v2", 64, 64, 3),
    ("v0", 128, 64, 3),
    ("v1", 128, 64, 3),
    ("v2", 128, 64, 3),
    ("v0", 256, 64, 3),
    ("v1", 256, 64, 3),
    ("v2", 256, 64, 3),
    ("v0", 512, 64, 3),
    ("v1", 512, 64, 3),
    ("v2", 512, 64, 3),
    ("v0", 1024, 64, 3),
    ("v1", 1024, 64, 3),
    ("v2", 1024, 64, 3),
]

def run_config(version, L, d, repeats):
    guest_cmd = f"cd /home/emu/lpu-workspace/lpu-bench/kernels/decode_attn && MAPS_DUMP_GPU_DEBUG_INFO_ENABLE=false MALOG_LEVEL=error ./decode_attn.out {version} {L} {d} {repeats}"
    full_cmd = f"{GUEST_RUN} '{guest_cmd}'"
    print(f"\n[RUN] Executing: {version}, L={L}, d={d}, repeats={repeats} ...", flush=True)
    
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
            if sline.startswith("kernel,") or sline.startswith("decode_attn,"):
                csv_lines.append(sline)

    return csv_lines

def main():
    parser = argparse.ArgumentParser(description="Run Decode Attention benchmarks")
    parser.add_argument("--filter-version", type=str, help="Comma-separated versions to run, e.g. v1")
    parser.add_argument("--filter-l", type=str, help="Comma-separated L values to run, e.g. 256,512,1024")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing decode_attn.csv")
    args = parser.parse_args()

    filter_versions = args.filter_version.split(",") if args.filter_version else None
    filter_ls = [int(x) for x in args.filter_l.split(",")] if args.filter_l else None

    configs_to_run = []
    for ver, l, d, reps in ALL_CONFIGS:
        if filter_versions and ver not in filter_versions:
            continue
        if filter_ls and l not in filter_ls:
            continue
        configs_to_run.append((ver, l, d, reps))

    build_id = get_binary_md5(BINARY_PATH)
    commit_hash = get_git_commit()

    print("==========================================================")
    print("      LPU-Bench: K2 Decode Attention Benchmark Suite       ")
    print("==========================================================")
    print(f"Binary Path: {BINARY_PATH}")
    print(f"Build ID (MD5): {build_id}")
    print(f"Git Commit: {commit_hash}")
    print(f"Configurations to run: {len(configs_to_run)}")
    
    write_header = args.overwrite or not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    header = None

    for version, L, d, repeats in configs_to_run:
        output_lines = run_config(version, L, d, repeats)
        if not output_lines:
            print(f"[ERROR] No output for {version} L={L}", file=sys.stderr)
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

    print("\n[COMPLETE] K2 Benchmark finished! Results written to:", RESULTS_CSV)

if __name__ == "__main__":
    main()
