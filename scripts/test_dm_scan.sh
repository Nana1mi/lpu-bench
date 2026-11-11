#!/bin/bash
set -e

DIR="/home/emu/lpu-workspace/lpu-bench/kernels/decode_attn"
cd "$DIR"

echo "=========================================================="
echo "    Scanning DM Capacity Beyond 512 (Up to 2048)          "
echo "=========================================================="

for L in 768 1024 1536 2048; do
    BYTES=$((L * 4 * 64))
    KB=$((BYTES / 1024))
    echo ""
    echo ">>> Testing RESIDENT_L=$L (K_resident size: ${KB} KB, V_resident size: ${KB} KB)..."

    # Compile with MAX_L equal to L
    SCALE_ARCH=$(basename $(dirname $(ls /opt/maps/toolchain/bin/*/clang++ 2>/dev/null | head -n 1)))
    set +e
    COMPILE_OUT=$(make clean >/dev/null 2>&1 && /opt/maps/toolchain/bin/${SCALE_ARCH}/clang++ decode_attn.sc -c -o decode_attn.o \
        -DRESIDENT_L=$L -DMAX_L=$L -DLPU_ARCH_TOKEN=${SCALE_ARCH} \
        -Xarch_host -I/opt/maps/include -Xarch_host -I/usr/include/c++/11 -Xarch_host -I/usr/include/x86_64-linux-gnu/c++/11/ -Xarch_host -I/usr/include/x86_64-linux-gnu/ -Xarch_host -I/usr/include/c++/11/backward -Xarch_host -I/opt/maps/toolchain/include/scale_include/llvm -Xarch_host -I/opt/maps/toolchain/include/scale_include/${SCALE_ARCH}-lib -Xarch_host -I../../common \
        -Xarch_device -I/opt/maps/include -Xarch_device -I/opt/maps/toolchain/include/scale_include/c++/v1 -Xarch_device -I/opt/maps/toolchain/include/scale_include/newlib -Xarch_device -I/opt/maps/toolchain/include/scale_include/llvm -Xarch_device -I/opt/maps/toolchain/include/scale_include/${SCALE_ARCH}-lib -Xarch_device -I../../common \
        -I/opt/maps/toolchain/include/scale_include -I/opt/maps/include -I../../common \
        -Xarch_device -nostdlibinc -std=c++11 -Xclang -target-sdk-version=9.2 -g -fdynamic-memory-base -DUSE_LLCDSC=1 2>&1)
    COMPILE_STATUS=$?
    set -e

    if [ $COMPILE_STATUS -ne 0 ]; then
        echo "  [COMPILE FAILED] Exit code $COMPILE_STATUS"
        echo "  Output: $COMPILE_OUT"
        continue
    fi
    echo "  [COMPILE OK]"

    # Link
    set +e
    LINK_OUT=$(/opt/maps/toolchain/bin/${SCALE_ARCH}/clang++ --scale-link -Wl,--check-scale-mem -o decode_attn.out decode_attn.o \
        -lscalehost -lmaps -lmapskmd -lllcdsc_kernel -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
        -L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L/opt/maps/toolchain/lib/scale/release -L/opt/maps/lib -L/lib/gcc/x86_64-linux-gnu/11/ -L/usr/lib/x86_64-linux-gnu/ 2>&1)
    LINK_STATUS=$?
    set -e

    if [ $LINK_STATUS -ne 0 ]; then
        echo "  [LINK FAILED] Exit code $LINK_STATUS"
        echo "  Output: $LINK_OUT"
        continue
    fi
    echo "  [LINK OK]"

    # Run execution test
    set +e
    RUN_OUT=$(MAPS_DUMP_GPU_DEBUG_INFO_ENABLE=false MALOG_LEVEL=error ./decode_attn.out v1 $L 64 1 2>&1)
    RUN_STATUS=$?
    set -e

    if [ $RUN_STATUS -ne 0 ]; then
        echo "  [RUN FAILED] Exit code $RUN_STATUS"
        echo "  Output: $RUN_OUT"
    else
        echo "  [RUN OK] Output snippet:"
        echo "$RUN_OUT" | grep -E "PASS|cycles|FAIL|Error" | head -n 5
    fi
done
