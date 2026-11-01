#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUEST_RUN = os.path.join(PROJECT_ROOT, "scripts", "guest_run.sh")
RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "decode_attn.csv")

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"

def get_timestamp():
    return datetime.datetime.now().isoformat()

os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

CONFIGS = [
    # (version, L, d, repeats)
    ("v0", 64, 64, 3),
    ("v1", 64, 64, 3),
    ("v2", 64, 64, 3),
    ("v0", 128, 64, 3),
    ("v1", 128, 64, 3),
    ("v2", 128, 64, 3),
    ("v0", 256, 64, 3),
    ("v2", 256, 64, 3),
    ("v0", 512, 64, 3),
    ("v2", 512, 64, 3),
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
    print("==========================================================")
    print("      LPU-Bench: K2 Decode Attention Benchmark Suite       ")
    print("==========================================================")
    
    write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    all_rows = []
    header = None

    for version, L, d, repeats in CONFIGS:
        output_lines = run_config(version, L, d, repeats)
        if not output_lines:
            print(f"[ERROR] No output for {version} L={L}", file=sys.stderr)
            continue
            
        cur_header = output_lines[0] + ",commit_hash,timestamp"
        if header is None:
            header = cur_header
            if write_header:
                with open(RESULTS_CSV, "w", newline="") as f:
                    f.write(header + "\n")
                write_header = False

        data_rows = output_lines[1:]
        commit_hash = get_git_commit()
        timestamp = get_timestamp()
        with open(RESULTS_CSV, "a", newline="") as f:
            for row in data_rows:
                out_row = f"{row},{commit_hash},{timestamp}"
                f.write(out_row + "\n")
                print(f"  -> Recorded: {out_row}")

    print("\n[COMPLETE] K2 Benchmark finished! Results written to:", RESULTS_CSV)

if __name__ == "__main__":
    main()
