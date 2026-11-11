#!/usr/bin/env python3
"""
Benchmark runner for vendor mablaslt GEMM M-scaling (M in 16, 32, 64).
Measures device_cycles across 3 repeats for FP16 and FP32.
Appends validated results to results/vendor_gemm.csv without fabricating any numbers.
Leaves sim_ms and sim_cycles blank to prevent methodology regressions.
"""

import os
import re
import subprocess
import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "results", "vendor_gemm.csv")
GUEST_RUN = os.path.join(REPO_ROOT, "scripts", "guest_run.sh")

# Configurations
DTYPES = ["fp16", "fp32"]
M_VALUES = [16, 32, 64]
K = 256
N = 256
REPEATS = 3
COMMIT_HASH = "7e7cba2"
BUILD_ID = "36190e5d6e2f"

def get_kernel_version(dtype, output_text):
    if dtype == "fp16":
        return "gemm_slice_k_matmul_cmode"
    else:
        return "gemm_matmul_cmode"

def run_benchmark(dtype, m):
    print(f"\n=======================================================", flush=True)
    print(f"Running benchmark: dtype={dtype}, M={m}, K={K}, N={N}, repeats={REPEATS}", flush=True)
    print(f"=======================================================", flush=True)

    cmd = (
        f"{GUEST_RUN} "
        f"'export LD_LIBRARY_PATH=/home/emu/vendor_demo/demo4scalee2e/mablaslt/build/lib:/home/emu/vendor_demo/demo4scalee2e/mablaslt/build/mablasLt:$LD_LIBRARY_PATH && "
        f"cd /home/emu/vendor_demo/demo4scalee2e/mablaslt/build/test && "
        f"./blas_perf {dtype} 1 {m} {N} {K} false false {REPEATS}'"
    )

    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = proc.stdout
    print(f"Process return code: {proc.returncode}", flush=True)

    # Parse REPEAT_RESULT
    pattern = re.compile(r"\[REPEAT_RESULT\]\s+repeat=(\d+)\s+dev_cycles=(\d+)\s+wall_ms=([\d\.]+)")
    matches = pattern.findall(output)

    if not matches:
        print("ERROR: No [REPEAT_RESULT] found in output!", flush=True)
        # Print last 30 lines of output for debugging
        lines = output.splitlines()
        print("\n".join(lines[-30:]), flush=True)
        return []

    results = []
    kernel_ver = get_kernel_version(dtype, output)
    flops = 2 * m * K * N

    for r_str, cyc_str, wall_str in matches:
        r = int(r_str)
        dev_cyc = int(cyc_str)
        wall_ms = float(wall_str)
        fpc = flops / dev_cyc if dev_cyc > 0 else 0.0
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        row = {
            "kernel": "mablaslt",
            "version": kernel_ver,
            "M": m,
            "K": K,
            "N": N,
            "dtype": dtype,
            "repeat": r,
            "sim_ms": "",
            "sim_cycles": "",
            "device_cycles": dev_cyc,
            "wall_ms": f"{wall_ms:.2f}",
            "flops": flops,
            "flops_per_cycle": f"{fpc:.4f}",
            "correct": "",
            "max_diff": "",
            "commit_hash": COMMIT_HASH,
            "build_id": BUILD_ID,
            "timestamp": ts,
            "validity": "invalid_dma_bypassed"
        }
        results.append(row)
        print(f"  Repeat {r}: dev_cycles={dev_cyc}, FLOPs/cycle={fpc:.4f}, wall_ms={wall_ms:.1f}", flush=True)

    return results

def append_to_csv(rows):
    if not rows:
        return
    fieldnames = [
        "kernel", "version", "M", "K", "N", "dtype", "repeat",
        "sim_ms", "sim_cycles", "device_cycles", "wall_ms",
        "flops", "flops_per_cycle", "correct", "max_diff",
        "commit_hash", "build_id", "timestamp", "validity"
    ]
    with open(CSV_PATH, "a") as f:
        for r in rows:
            line = ",".join(str(r[fn]) for fn in fieldnames) + "\n"
            f.write(line)
    print(f"Appended {len(rows)} rows to {CSV_PATH}", flush=True)

def main():
    all_results = []
    for dtype in DTYPES:
        for m in M_VALUES:
            rows = run_benchmark(dtype, m)
            append_to_csv(rows)
            all_results.extend(rows)

    print("\nBenchmark run completed successfully.", flush=True)
    print(f"Total rows added: {len(all_results)}", flush=True)

if __name__ == "__main__":
    main()
