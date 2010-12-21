CUDA_PATH = /usr/local/cuda

# compilers
CC := gcc
CXX = g++
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_DBG_FLAGS = -Wall -Xcompiler -fno-strict-aliasing,-fPIC

INCPATH =
# compiler / linker flags
CCFLAGS = -fPIC -g -Wall

# linker flags
LINKFLAGS = -L. -lgpustats

LIBPATH =
NVCCFLAGS = $(NVCC_DBG_FLAGS)

OSUPPER   = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER   = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
OSNAME := $(shell uname)
OSARCH = $(shell uname -m)

ifeq ($(OSNAME),Linux)
	CUDA_SDK_PATH   := $(HOME)/cuda_sdk
	CUDA_LIB  = -L$(CUDA_PATH)/lib64
endif

# OS X thinks it's i386
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

ifeq ($(OSNAME),Darwin)
	CUDA_SDK_PATH   := /Developer/CUDA/C
	CUDA_LIB  = -L$(CUDA_PATH)/lib
endif

CC_ARCH_FLAGS :=
NVCCFLAGS :=

ifeq ($(OSNAME),Darwin)
	NVCCFLAGS += -m32
	LIB_ARCH = i386
	CC_ARCH_FLAGS += -arch i386
else
	LIB_ARCH        = x86_64
	CUDPPLIB_SUFFIX = x86_64
	NVCCFLAGS      += -m64
	ifneq ($(DARWIN),)
	   CXX_ARCH_FLAGS += -arch x86_64
	else
	   CXX_ARCH_FLAGS += -m64
	endif
endif

CCFLAGS += $(CC_ARCH_FLAGS)

CUDA_INC = -I$(CUDA_PATH)/include
CUDA_SDK_COMMONDIR = $(CUDA_SDK_PATH)/common
CUDA_SDK_INC = -I$(CUDA_SDK_COMMONDIR)/inc
CUDA_LIB += -L$(CUDA_SDK_PATH)/lib -L$(CUDA_SDK_COMMONDIR)/lib -lcuda -lcudart -lcublas

EXECUTABLE := test
CUFILES = mvnpdf.cu
CU_DEPS = common.h
CFILES := common.c
USECUBLAS        := 1
OBJDIR = obj
LIBDIR = lib
TARGETDIR = .
TARGET    := $(TARGETDIR)/$(EXECUTABLE)

OBJS +=  $(patsubst %.c,%.o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,%.cu_o,$(notdir $(CUFILES)))

VERBOSE := -

# need to use g++ to link on OS X?

libgpustats.so: makedirs $(OBJS)
	$(CXX) $(CC_ARCH_FLAGS) -shared -W1,-soname,libgpustats.so -o libgpustats.so $(OBJS) -lc $(CUDA_LIB)

runpy: cython
	LD_LIBRARY_PATH=.:$(LD_LIBRARY_PATH)  python scratch.py

test: libgpustats.so
	$(VERBOSE)$(CC) $(CC_ARCH_FLAGS) -std=c99 test.c -o test $(CUDA_INC) $(LINKFLAGS)

cython: libgpustats.so cytest.pyx build_cython.py
	-python build_cython.py build_ext --inplace

makedirs:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(TARGETDIR)

clean:
	-rm -rf *.so *.o *.cu_o build/

#### CUDA files

%.o: %.c
	$(VERBOSE)$(CC) $(CCFLAGS) -c $*.c -o $@ $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC)

%.c_o : %.c
	$(CC) $(PROFILE) -c $< -o $@

%.cu_o : %.cu $(CUDA_HEADERS) $(CU_DEPS)
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) -c $< -o $@ -I. $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC) -DUNIX
