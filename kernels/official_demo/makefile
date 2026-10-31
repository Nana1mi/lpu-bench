# set up default (release) environment
MAPS_DIR ?= /opt/maps
INC_MAPS ?= $(MAPS_DIR)/include
LIB_MAPS ?= $(MAPS_DIR)/lib
LIB_MAPSKMD ?= $(MAPS_DIR)/lib
SCALE_ARCH ?= aurora

TOOLCHAIN_DIR ?= /opt/maps/toolchain
ifeq ($(SCALE_ARCH), aurora)
	INC_TC := $(TOOLCHAIN_DIR)/include/scale_include
	LIB_TC := $(TOOLCHAIN_DIR)/lib/scale/release
	LIB_LLVM := $(TOOLCHAIN_DIR)/lib/scale/llvm_dynamic
	TC_BIN := $(TOOLCHAIN_DIR)/bin/aurora
else
	INC_TC := $(TOOLCHAIN_DIR)/include/prj_s_scale_include
	LIB_TC := $(TOOLCHAIN_DIR)/lib/prj_s-scale/release
	LIB_LLVM := $(TOOLCHAIN_DIR)/lib/prj_s-scale/llvm_dynamic
	TC_BIN := $(TOOLCHAIN_DIR)/bin/prj_s
endif

LLCDSC_DIR ?= /opt/maps
ifneq ($(wildcard $(LLCDSC_DIR)/inc),)
	INC_LLCDSC := $(LLCDSC_DIR)/inc
else
	INC_LLCDSC := $(LLCDSC_DIR)/include
endif

export PATH := $(TC_BIN):$(PATH)

GCC_MAJOR:=$(shell gcc -dumpfullversion | cut -d. -f1)
GCC_MINOR:=$(shell gcc -dumpfullversion | cut -d. -f2)
GLIBC_MAJOR_CMD:=`awk '$$1 ~ /\#define/ && $$2 ~ /__GLIBC__/ {printf "%d", $$3}' /usr/include/features.h`
GLIBC_MAJOR:=$(shell echo -n $(GLIBC_MAJOR_CMD))
GLIBC_MINOR_CMD:=`awk '$$1 ~ /\#define/ && $$2 ~ /__GLIBC_MINOR__/ {printf "%d", $$3}' /usr/include/features.h`
GLIBC_MINOR:=$(shell echo -n $(GLIBC_MINOR_CMD))

# device
DEVICE_INC=$(INC_TC)/c++/v1
DEVICE_INC2=$(INC_TC)/newlib
DEVICE_INC3=$(INC_TC)/llvm
ifeq ($(SCALE_ARCH), aurora)
	DEVICE_INC4=$(INC_TC)/aurora-lib
else
	DEVICE_INC4=$(INC_TC)/prj_s-lib
endif

# host
ifeq ($(GCC_MAJOR).$(GCC_MINOR),8.3)
	HOST_INC=/tool/devtoolset/8.3.1/root/usr/include/c++/8
	HOST_INC2=/tool/devtoolset/8.3.1/root/usr/include/c++/8/x86_64-redhat-linux
	HOST_INC3=/tool/devtoolset/8.3.1/root/usr/include/c++/8/x86_64-redhat-linux
	HOST_INC4=/tool/devtoolset/8.3.1/root/usr/include/c++/8/backward/
	HOST_LIB=/tool/devtoolset/8.3.1/root/usr/lib/gcc/x86_64-redhat-linux/8
	HOST_LIB2=/usr/lib64/
else
	HOST_INC=/usr/include/c++/11
	HOST_INC2=/usr/include/x86_64-linux-gnu/c++/11/
	HOST_INC3=/usr/include/x86_64-linux-gnu/
	HOST_INC4=/usr/include/c++/11/backward
	HOST_LIB=/lib/gcc/x86_64-linux-gnu/11/
	HOST_LIB2=/usr/lib/x86_64-linux-gnu/
endif

HOST_INC5=$(INC_TC)/llvm
HOST_LIB3=$(LLCDSC_DIR)
HOST_LIB4=$(LLCDSC_DIR)/lib

ifeq ($(SCALE_ARCH), aurora)
	ARG_LLCDSC1 = -DUSE_LLCDSC=1
	ARG_LLCDSC2 = -lllcdsc_kernel
	ARG_PROJECTS =
else
	ARG_LLCDSC1 = -DUSE_LLCDSC=0
	ARG_LLCDSC2 =
	ARG_PROJECTS = -fis-projectS
endif

ifeq ($(GLIBC_MAJOR).$(GLIBC_MINOR),2.17)
	FLAGS=-std=c++11 -Xclang -target-sdk-version=9.2 -g -fdynamic-memory-base $(ARG_PROJECTS) $(ARG_LLCDSC1) -fnew-alignment=8
else
	FLAGS=-std=c++11 -Xclang -target-sdk-version=9.2 -g -fdynamic-memory-base $(ARG_PROJECTS) $(ARG_LLCDSC1)
endif

all:
	$(TC_BIN)/clang++ apc_perf_test.sc -c \
	-Xarch_host -I$(INC_MAPS) -Xarch_host -I$(HOST_INC) -Xarch_host -I$(HOST_INC2) -Xarch_host -I$(HOST_INC3) -Xarch_host -I$(HOST_INC4) -Xarch_host -I$(HOST_INC5) \
	-Xarch_device -I$(INC_MAPS) -Xarch_device -I$(DEVICE_INC) -Xarch_device -I$(DEVICE_INC2) -Xarch_device -I$(DEVICE_INC3) -Xarch_device -I$(DEVICE_INC4) \
	-I$(INC_TC) -I$(INC_LLCDSC) \
	-Xarch_device -nostdlibinc $(FLAGS)

	$(TC_BIN)/clang++ process_device.sc -c \
	-Xarch_host -I$(INC_MAPS) -Xarch_host -I$(HOST_INC) -Xarch_host -I$(HOST_INC2) -Xarch_host -I$(HOST_INC3) -Xarch_host -I$(HOST_INC4) -Xarch_host -I$(HOST_INC5) \
	-Xarch_device -I$(INC_MAPS) -Xarch_device -I$(DEVICE_INC) -Xarch_device -I$(DEVICE_INC2) -Xarch_device -I$(DEVICE_INC3) -Xarch_device -I$(DEVICE_INC4) \
	-I$(INC_TC) -I$(INC_LLCDSC) \
	-Xarch_device -nostdlibinc $(FLAGS)

	$(TC_BIN)/clang++ main.sc -c \
	-Xarch_host -I$(INC_MAPS) -Xarch_host -I$(HOST_INC) -Xarch_host -I$(HOST_INC2) -Xarch_host -I$(HOST_INC3) -Xarch_host -I$(HOST_INC4) -Xarch_host -I$(HOST_INC5) \
	-Xarch_device -I$(INC_MAPS) -Xarch_device -I$(DEVICE_INC) -Xarch_device -I$(DEVICE_INC2) -Xarch_device -I$(DEVICE_INC3) -Xarch_device -I$(DEVICE_INC4) \
	-I$(INC_TC) -I$(INC_LLCDSC) \
	-Xarch_device -nostdlibinc $(FLAGS)

	$(TC_BIN)/clang++ tenK_end2end.sc -c \
	-Xarch_host -I$(INC_MAPS) -Xarch_host -I$(HOST_INC) -Xarch_host -I$(HOST_INC2) -Xarch_host -I$(HOST_INC3) -Xarch_host -I$(HOST_INC4) -Xarch_host -I$(HOST_INC5) \
	-Xarch_device -I$(INC_MAPS) -Xarch_device -I$(DEVICE_INC) -Xarch_device -I$(DEVICE_INC2) -Xarch_device -I$(DEVICE_INC3) -Xarch_device -I$(DEVICE_INC4) \
	-I$(INC_TC) -I$(INC_LLCDSC) \
	-Xarch_device -nostdlibinc $(FLAGS)

	$(TC_BIN)/clang++ perf.sc -c \
	-Xarch_host -I$(INC_MAPS) -Xarch_host -I$(HOST_INC) -Xarch_host -I$(HOST_INC2) -Xarch_host -I$(HOST_INC3) -Xarch_host -I$(HOST_INC4) -Xarch_host -I$(HOST_INC5) \
	-Xarch_device -I$(INC_MAPS) -Xarch_device -I$(DEVICE_INC) -Xarch_device -I$(DEVICE_INC2) -Xarch_device -I$(DEVICE_INC3) -Xarch_device -I$(DEVICE_INC4) \
	-I$(INC_TC) -I$(INC_LLCDSC) \
	-Xarch_device -nostdlibinc $(FLAGS)

	$(TC_BIN)/clang++ traverse_device.sc -c \
	-Xarch_host -I$(INC_MAPS) -Xarch_host -I$(HOST_INC) -Xarch_host -I$(HOST_INC2) -Xarch_host -I$(HOST_INC3) -Xarch_host -I$(HOST_INC4) -Xarch_host -I$(HOST_INC5) \
	-Xarch_device -I$(INC_MAPS) -Xarch_device -I$(DEVICE_INC) -Xarch_device -I$(DEVICE_INC2) -Xarch_device -I$(DEVICE_INC3) -Xarch_device -I$(DEVICE_INC4) \
	-I$(INC_TC) -I$(INC_LLCDSC) \
	-Xarch_device -nostdlibinc $(FLAGS)

	$(TC_BIN)/clang++ kernels.sc -c \
	-Xarch_host -I$(DEVICE_INC4) \
	-Xarch_host -I$(INC_MAPS) -Xarch_host -I$(HOST_INC) -Xarch_host -I$(HOST_INC2) -Xarch_host -I$(HOST_INC3) -Xarch_host -I$(HOST_INC4) -Xarch_host -I$(HOST_INC5) \
	-Xarch_device -I$(INC_MAPS) -Xarch_device -I$(DEVICE_INC) -Xarch_device -I$(DEVICE_INC2) -Xarch_device -I$(DEVICE_INC3) -Xarch_device -I$(DEVICE_INC4) \
	-I$(INC_TC) -I$(INC_LLCDSC) \
	-Xarch_device -nostdlibinc $(FLAGS)

ifeq ($(SCALE_ARCH), aurora)
	$(TC_BIN)/clang++ -c -scale-mpu-asm mpu_tanh_fp32_a.m.asm -o 2.o $(FLAGS)
else
	$(TC_BIN)/clang++ -c -scale-mpu-asm mpu_tanh_fp32_s.m.asm -o 2.o $(FLAGS)
endif

	$(TC_BIN)/clang++ -c -scale-mpu-asm mpu_add_u32.m.asm -o mpu_add_u32.o $(FLAGS)

ifeq ($(GCC_MAJOR).$(GCC_MINOR),8.3)
	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o apc_perf_test.out 2.o apc_perf_test.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L$(HOST_LIB) -L$(HOST_LIB2) -L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o a.out 2.o mpu_add_u32.o main.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L$(HOST_LIB) -L$(HOST_LIB2) -L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB3) -L$(HOST_LIB4)
	
	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o tenK_end2end.out 2.o tenK_end2end.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L$(HOST_LIB) -L$(HOST_LIB2) -L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o perf.out 2.o mpu_add_u32.o perf.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L$(HOST_LIB) -L$(HOST_LIB2) -L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o traverse_device.out 2.o traverse_device.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L$(HOST_LIB) -L$(HOST_LIB2) -L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o process_device.out 2.o process_device.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L$(HOST_LIB) -L$(HOST_LIB2) -L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB3) -L$(HOST_LIB4)
else
	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o apc_perf_test.out 2.o apc_perf_test.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB) -L$(HOST_LIB2) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o a.out 2.o mpu_add_u32.o main.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB) -L$(HOST_LIB2) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o tenK_end2end.out 2.o tenK_end2end.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB) -L$(HOST_LIB2) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o perf.out 2.o mpu_add_u32.o perf.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB) -L$(HOST_LIB2) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o traverse_device.out 2.o traverse_device.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB) -L$(HOST_LIB2) -L$(HOST_LIB3) -L$(HOST_LIB4)

	$(TC_BIN)/clang++ --scale-link $(ARG_PROJECTS) -o process_device.out 2.o process_device.o kernels.o \
	-lscalehost -lmaps -lmapskmd $(ARG_LLCDSC2) -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB) -L$(HOST_LIB2) -L$(HOST_LIB3) -L$(HOST_LIB4)
endif

clean:
	rm -rf *.o *.out
