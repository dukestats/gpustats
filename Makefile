CC := gcc
CXX = g++
LINK = g++ -fPIC -m64
CXXFLAGS = -fPIC -m64
INCPATH =
LIBPATH =
OSNAME := $(shell uname)

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
CFILES := gpustats_common.c test.c
USECUBLAS        := 1
OBJDIR = obj
LIBDIR = lib
TARGETDIR = .
TARGET    := $(TARGETDIR)/$(EXECUTABLE)

OBJS +=  $(patsubst %.c,%.o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,%.cu_o,$(notdir $(CUFILES)))

VERBOSE := -

output: makedirs $(OBJS)
	$(VERBOSE)$(LINK) -o $(TARGET) $(OBJS) $(CUDA_LIB)

makedirs:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(TARGETDIR)

clean :
	-rm -f *.o
	-rm -f *.cu_o

#### CUDA files

%.o: %.c
	$(VERBOSE)$(CXX) $(CXXFLAGS) -c $*.c -o $@ $(INCPATH) $(CUDA_INC) $(CUDA_SDK_INC)

%.c_o : %.c
	$(GCC) $(PROFILE) -c $< -o $@ $(LOCAL_INC)

%.cpp_o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCPATH) $(LOCAL_INC) $(CUDA_INC) $(CUDA_SDK_INC) $(PROFILE) -c $< -o $@

%.cu_o : %.cu $(CUDA_HEADERS)
	$(VERBOSE)$(NVCC) $(NVCC_DBG_FLAGS) -c $< -o $@ -I. $(INCPATH) $(LOCAL_INC) $(CUDA_INC) $(CUDA_SDK_INC) -DUNIX
