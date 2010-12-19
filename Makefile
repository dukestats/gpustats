CC := gcc
CXX = g++
CCLINK = gcc -shared -W1
CCFLAGS = -fPIC -g -Wall
CXXFLAGS = -fPIC -m64
INCPATH =
LIBPATH =
OSNAME := $(shell uname)

LFLAGS = -L. -lgpustats

CUDA_SDK_PATH   := $(HOME)/cuda_sdk
CUDA_PATH = /usr/local/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_SDK_COMMONDIR = $(CUDA_SDK_PATH)/common
CUDA_SDK_INC = -I$(CUDA_SDK_COMMONDIR)/inc
CUDA_LIB  = -L$(CUDA_PATH)/lib64 -L$(CUDA_SDK_PATH)/lib -L$(CUDA_SDK_COMMONDIR)/lib -lcuda -lcudart -lcublas
NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_DBG_FLAGS = -Xcompiler -fno-strict-aliasing,-fPIC

EXECUTABLE := test
CUFILES = mvnpdf.cu
CU_DEPS = gpustats_common.h
CFILES := gpustats_common.c
USECUBLAS        := 1
OBJDIR = obj
LIBDIR = lib
TARGETDIR = .
TARGET    := $(TARGETDIR)/$(EXECUTABLE)

OBJS +=  $(patsubst %.c,%.o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,%.cu_o,$(notdir $(CUFILES)))

VERBOSE := -

libgpustats.so: makedirs $(OBJS)
	gcc -shared -W1,-soname,libgpustats.so -o libgpustats.so $(OBJS) -lc $(CUDA_LIB)

runpy: libgpustats.so
    LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH; python scratch.py

test: libgpustats.so
	$(VERBOSE)$(CC) -std=c99 test.c -o test $(CUDA_INC) $(LFLAGS)

cython: libgpustats.so
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

%.cu_o : %.cu $(CUDA_HEADERS)
	$(VERBOSE)$(NVCC) $(NVCC_DBG_FLAGS) -c $< -o $@ -I. $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC) -DUNIX
