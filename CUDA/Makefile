all:	gpuMinReduction.ptx gpuSelectionSort.ptx gpuWDLA.ptx
CUDA_PATH       := /usr/local/cuda-5.0
CUDA_INC_PATH   := $(CUDA_PATH)/include
CUDA_BIN_PATH   := $(CUDA_PATH)/bin
CUDA_LIB_PATH   := $(CUDA_PATH)/lib64
NVCC            := $(CUDA_BIN_PATH)/nvcc
GCC             := g++

GENCODE_FLAGS :=  -dc -gencode arch=compute_35,code=sm_35
LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
CCFLAGS   := -m 64
NVCCFLAGS := -m 64
INCLUDES  := -I$(CUDA_INC_PATH) -I. -I$(CUDA_PATH)/samples/common/inc

gpuSelectionSort.ptx: gpuSelectionSort.cu  
	$(NVCC) -ptx $(NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) $<
gpuWDLA.ptx: gpuWDLA.cu
	$(NVCC) -ptx $(NVCCFLAGS) $(INCLUDES) $<
gpuMinReduction.ptx: gpuMinReduction.cu
	$(NVCC) -ptx $(NVCCFLAGS) $(INCLUDES) $<
