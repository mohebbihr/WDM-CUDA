#include <limits.h>

#define NumThread 128
#define NumBlock 32

__global__ void min_reduce(int* In, int* Out, int * OutIdx, int n){
  __shared__ int sdata[NumThread];
  __shared__ int sIdxdata[NumThread];
  unsigned int i = blockIdx.x * NumThread + threadIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = NumBlock * NumThread;
  int myMin = INT_MAX;
  int myMinIdx = -1;

  while (i < n){
    if(In[i] < myMin){
    	myMin = In[i];
	myMinIdx = i;
    }
    i += gridSize;
  }
  sdata[tid] = myMin;
  sIdxdata[tid] = myMinIdx;
  __syncthreads();

  if (NumThread >= 1024){
    if (tid < 512)
    if(sdata[tid] > sdata[tid + 512] ){ sdata[tid] = sdata[tid + 512]; sIdxdata[tid] = sIdxdata[tid + 512]; }
    __syncthreads();
  }
  if (NumThread >= 512){
    if(sdata[tid] > sdata[tid + 256] ){ sdata[tid] = sdata[tid + 256]; sIdxdata[tid] = sIdxdata[tid + 256]; }
    __syncthreads();
  }
  if (NumThread >= 256){
    if(sdata[tid] > sdata[tid + 128] && sdata[tid + 128] !=0){ sdata[tid] =  sdata[tid + 128]; sIdxdata[tid] = sIdxdata[tid + 128]; }
    __syncthreads();
  }
  if (NumThread >= 128){
    if (tid < 64)
    if(sdata[tid] > sdata[tid + 64] ){ sdata[tid] =    sdata[tid + 64]; sIdxdata[tid] = sIdxdata[tid + 64]; }
    __syncthreads();
  }
  //the following practice is deprecated
   if (tid < 32){
    volatile int *smem = sdata;
    volatile int *sidx = sIdxdata;

    if (NumThread >= 64) if(smem[tid] > smem[tid + 32]){ smem[tid] =  smem[tid+32]; sidx[tid] =  sidx[tid+32]; }
    if (NumThread >= 32) if(smem[tid] > smem[tid + 16]){ smem[tid] =  smem[tid+16]; sidx[tid] =  sidx[tid+16]; }
    if (NumThread >= 16) if(smem[tid] > smem[tid + 8]){ smem[tid] =  smem[tid+8]; sidx[tid] =  sidx[tid+8]; }
    if (NumThread >= 8) if(smem[tid] > smem[tid + 4]){ smem[tid] =  smem[tid+4]; sidx[tid] =  sidx[tid+4]; }
    if (NumThread >= 4) if(smem[tid] > smem[tid + 2]){ smem[tid] =  smem[tid+2]; sidx[tid] =  sidx[tid+2]; }
    if (NumThread >= 2) if(smem[tid] > smem[tid + 1]){ smem[tid] =  smem[tid+1]; sidx[tid] =  sidx[tid+1]; }
  }
  if (tid == 0){
    if(sdata[0] < sdata[1] ){ Out[blockIdx.x] = sdata[0]; OutIdx[blockIdx.x] = sIdxdata[0];}
    else{ Out[blockIdx.x] = sdata[1]; OutIdx[blockIdx.x] = sIdxdata[1];}
  }
}




