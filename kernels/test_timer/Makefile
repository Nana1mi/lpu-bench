MAPS_DIR ?= /opt/maps
INC_MAPS ?= $(MAPS_DIR)/include
LIB_MAPS ?= $(MAPS_DIR)/lib
LIB_MAPSKMD ?= $(MAPS_DIR)/lib
SCALE_ARCH ?= aurora

TOOLCHAIN_DIR ?= /opt/maps/toolchain
INC_TC := $(TOOLCHAIN_DIR)/include/scale_include
LIB_TC := $(TOOLCHAIN_DIR)/lib/scale/release
TC_BIN := $(TOOLCHAIN_DIR)/bin/aurora
INC_LLCDSC := $(MAPS_DIR)/include

DEVICE_INC := $(INC_TC)/c++/v1
DEVICE_INC2 := $(INC_TC)/newlib
DEVICE_INC3 := $(INC_TC)/llvm
DEVICE_INC4 := $(INC_TC)/aurora-lib

HOST_INC := /usr/include/c++/11
HOST_INC2 := /usr/include/x86_64-linux-gnu/c++/11/
HOST_INC3 := /usr/include/x86_64-linux-gnu/
HOST_INC4 := /usr/include/c++/11/backward
HOST_LIB := /lib/gcc/x86_64-linux-gnu/11/
HOST_LIB2 := /usr/lib/x86_64-linux-gnu/
HOST_INC5 := $(INC_TC)/llvm

INC_COMMON := ../../common

FLAGS := -std=c++11 -Xclang -target-sdk-version=9.2 -g -fdynamic-memory-base -DUSE_LLCDSC=1

all: test_timer.out

test_timer.o: test_timer.sc
	$(TC_BIN)/clang++ test_timer.sc -c \
	-Xarch_host -I$(INC_MAPS) -Xarch_host -I$(HOST_INC) -Xarch_host -I$(HOST_INC2) -Xarch_host -I$(HOST_INC3) -Xarch_host -I$(HOST_INC4) -Xarch_host -I$(HOST_INC5) -Xarch_host -I$(INC_COMMON) \
	-Xarch_device -I$(INC_MAPS) -Xarch_device -I$(DEVICE_INC) -Xarch_device -I$(DEVICE_INC2) -Xarch_device -I$(DEVICE_INC3) -Xarch_device -I$(DEVICE_INC4) -Xarch_device -I$(INC_COMMON) \
	-I$(INC_TC) -I$(INC_LLCDSC) -I$(INC_COMMON) \
	-Xarch_device -nostdlibinc $(FLAGS)

test_timer.out: test_timer.o
	$(TC_BIN)/clang++ --scale-link -o test_timer.out test_timer.o \
	-lscalehost -lmaps -lmapskmd -lllcdsc_kernel -lelf -lstdc++ -lm -lgcc -lc -lgcc_eh \
	-L/lib64 -L/usr/lib64 -L/usr/lib/x86_64-linux-gnu -L$(LIB_TC) -L$(LIB_MAPS) -L$(LIB_MAPSKMD) -L$(HOST_LIB) -L$(HOST_LIB2)

clean:
	rm -rf *.o *.out
