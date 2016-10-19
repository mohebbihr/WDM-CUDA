#include <stdio.h>
#include <stdlib.h>

#include "min_reduction.cu"

int main(int argc, char* argv[]){

  unsigned int length = 20; //1048576;
  int i, Size, min, minIdx;
  int *a, *out, *outIdx, *gpuA, *gpuOut, *gpuOutIdx;

  cudaSetDevice(0);
  Size = length * sizeof(int);
  a = (int*)malloc(Size);
  out = (int*)malloc(NumBlock*sizeof(int));
  outIdx = (int*)malloc(NumBlock*sizeof(int));
  for(i=0;i<length;i++) a[i] = (i + 10);

  a[10] = 5;

  cudaMalloc((void**)&gpuA,Size);
  cudaMalloc((void**)&gpuOut,NumBlock*sizeof(int));
  cudaMalloc((void**)&gpuOutIdx,NumBlock*sizeof(int));
  cudaMemcpy(gpuA,a,Size,cudaMemcpyHostToDevice);
  min_reduce<<<NumBlock,NumThread>>>(gpuA,gpuOut, gpuOutIdx, length);
  cudaDeviceSynchronize();
  cudaMemcpy(out,gpuOut,NumBlock*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(outIdx,gpuOutIdx,NumBlock*sizeof(int),cudaMemcpyDeviceToHost);
 
  //printf("out array \n");
  //for(i=0; i<NumBlock; i++) printf("out[%d] = %d\n", i, out[i]);
  //printf("\n");
  
  min = out[0];
  minIdx = outIdx[0];
  for(i=1;i<NumBlock;i++) if(min > out[i]){ min = out[i]; minIdx = outIdx[i]; }
  printf("min: %d, minIdx: %d \n",min, minIdx);
  return 0;
}
