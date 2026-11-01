#!/bin/bash

cat << 'EOF' > /tmp/test_mem.sc
#include <scale.h>

#define NUM_VECS 4

__global__ void test_k(int *out) {
#ifdef __aurora__
    __mem0__ v16fp32 buf[CAP_L * NUM_VECS] __attribute__((aligned(64)));
    buf[0][0] = 1.0f;
    *out = (int)buf[0][0];
#endif
}

int main() {
    return 0;
}
EOF

MAPS_DIR=/opt/maps
TOOLCHAIN_DIR=/opt/maps/toolchain
INC_MAPS=$MAPS_DIR/include
INC_TC=$TOOLCHAIN_DIR/include/scale_include
LIB_TC=$TOOLCHAIN_DIR/lib/scale/release
TC_BIN=$TOOLCHAIN_DIR/bin/aurora

DEVICE_INC=$INC_TC/c++/v1
DEVICE_INC2=$INC_TC/newlib
DEVICE_INC3=$INC_TC/llvm
DEVICE_INC4=$INC_TC/aurora-lib

HOST_INC=/usr/include/c++/11
HOST_INC2=/usr/include/x86_64-linux-gnu/c++/11/
HOST_INC3=/usr/include/x86_64-linux-gnu/
HOST_INC4=/usr/include/c++/11/backward
HOST_LIB=/lib/gcc/x86_64-linux-gnu/11/
HOST_LIB2=/usr/lib/x86_64-linux-gnu/
HOST_INC5=$INC_TC/llvm

FLAGS="-std=c++11 -Xclang -target-sdk-version=9.2 -g -fdynamic-memory-base -DUSE_LLCDSC=1"

for L in 128 160 192 224 256 384 512 1024; do
    echo "=========================================="
    echo "Testing CAP_L=$L (size=$((L * 4 * 64 / 1024)) KB)"
    if $TC_BIN/clang++ -DCAP_L=$L /tmp/test_mem.sc -c -o /tmp/test_mem.o \
        -Xarch_host -I$INC_MAPS -Xarch_host -I$HOST_INC -Xarch_host -I$HOST_INC2 -Xarch_host -I$HOST_INC3 -Xarch_host -I$HOST_INC4 -Xarch_host -I$HOST_INC5 -Xarch_host -I$DEVICE_INC4 \
        -Xarch_device -I$INC_MAPS -Xarch_device -I$DEVICE_INC -Xarch_device -I$DEVICE_INC2 -Xarch_device -I$DEVICE_INC3 -Xarch_device -I$DEVICE_INC4 \
        -I$INC_TC -I$INC_MAPS \
        -Xarch_device -nostdlibinc $FLAGS; then
        echo "COMPILE: OK"
        if $TC_BIN/clang++ --scale-link -Wl,--check-scale-mem -o /tmp/test_mem.out /tmp/test_mem.o \
            -lscalehost -lmaps -lmapskmd -lllcdsc_kernel -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
            -L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$LIB_TC -L$MAPS_DIR/lib -L$HOST_LIB -L$HOST_LIB2; then
            echo "LINK: OK"
        else
            echo "LINK: FAILED"
        fi
    else
        echo "COMPILE: FAILED"
    fi
done
