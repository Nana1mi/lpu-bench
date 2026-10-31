#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="/home/rog/miniconda3/bin/python"

if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=========================================================="
echo "      LPU-Bench: End-to-End Benchmark & Reproduction      "
echo "=========================================================="
echo "Project root: $PROJECT_ROOT"
echo "Using Python: $PYTHON_BIN"
echo ""

# Step 1: Health check guest VM
echo "[STEP 1/5] Checking QEMU CModel Guest Connection..."
"$SCRIPT_DIR/guest_run.sh" "uname -a && cat /proc/cpuinfo | grep 'model name' | head -n 1"
echo ">> Guest CModel connection healthy!"
echo ""

# Step 2: Build all benchmark kernels
echo "[STEP 2/5] Building benchmark binaries..."
"$SCRIPT_DIR/guest_run.sh" "cd /home/emu/lpu-workspace/lpu-bench/kernels/test_timer && make clean && make"
"$SCRIPT_DIR/guest_run.sh" "cd /home/emu/lpu-workspace/lpu-bench/kernels/gemm && make clean && make"
"$SCRIPT_DIR/guest_run.sh" "cd /home/emu/lpu-workspace/lpu-bench/kernels/decode_attn && make clean && make"
echo ">> All kernels compiled and linked successfully!"
echo ""

# Step 3: Run Benchmarks
echo "[STEP 3/5] Running benchmark suites..."
echo "--- Running K1 GEMM Scaling Sweep ---"
$PYTHON_BIN "$PROJECT_ROOT/scripts/run_k1_gemm.py"

echo "--- Running K2 Decode Attention Sweep ---"
$PYTHON_BIN "$PROJECT_ROOT/scripts/run_k2_attn.py"

echo "--- Running K3 Overlap & Locality Ablation ---"
$PYTHON_BIN "$PROJECT_ROOT/scripts/run_k3_ablation.py"
echo ""

# Step 4: Generate figures
echo "[STEP 4/5] Generating thesis evaluation figures..."
$PYTHON_BIN "$PROJECT_ROOT/scripts/plot.py"
echo ""

# Step 5: Verification & Summary
echo "[STEP 5/5] Benchmark deliverables verification:"
ls -lh "$PROJECT_ROOT/results"
ls -lh "$PROJECT_ROOT/figures"
echo ""
echo "=========================================================="
echo "      LPU-Bench: End-to-End Reproduction COMPLETE!        "
echo "=========================================================="
