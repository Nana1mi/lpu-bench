#!/usr/bin/env python3
import subprocess
import csv
import re
import os
import sys

def run_guest(cmd):
    full_cmd = ["/home/rog/lpu-workspace/lpu-bench/scripts/guest_run.sh", cmd]
    p = subprocess.run(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout

def main():
    modes = [
        (0, "fma_reg_unfused"),
        (1, "fma_reg_fused"),
        (2, "fma_dm_gemm_pattern")
    ]
    N_list = [100, 1000, 10000]
    repeats = 3

    results = []
    
    for mode_idx, mode_name in modes:
        for N in N_list:
            print(f"Running mode={mode_idx} ({mode_name}), N={N} ...", flush=True)
            cmd = f"/home/emu/lpu-workspace/lpu-bench/kernels/test_fma/test_fma.out {N} {mode_idx} {repeats}"
            out = run_guest(cmd)
            # Find RESULT: lines
            matches = re.findall(r"RESULT:\s+N=(\d+)\s+mode=(\d+)\s+repeat=(\d+)\s+cycles=(\d+)", out)
            if not matches:
                print(f"Warning: No match found in output for mode {mode_idx}, N {N}. Output tail:")
                print(out[-500:])
                continue
            
            for m in matches:
                n_val, m_val, rep, cyc = int(m[0]), int(m[1]), int(m[2]), int(m[3])
                results.append({
                    "mode_idx": m_val,
                    "mode_name": mode_name,
                    "N": n_val,
                    "repeat": rep,
                    "cycles": cyc
                })
                print(f"  -> rep {rep}: {cyc} cycles (avg {cyc / n_val:.2f} cyc/iter)")

    os.makedirs("/home/rog/lpu-workspace/lpu-bench/results", exist_ok=True)
    csv_path = "/home/rog/lpu-workspace/lpu-bench/results/test_fma.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode_idx", "mode_name", "N", "repeat", "cycles"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()
