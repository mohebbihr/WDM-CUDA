#define BlockSize 16
#define Tolerance 0.0001

__global__ void gpuEuclidean(const double *input, int numRow, int numCol, double *output){
  __shared__ float Ys[BlockSize][BlockSize];
  __shared__ float Xs[BlockSize][BlockSize]; 
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int yBegin = by * BlockSize * numCol;
  int xBegin = bx * BlockSize * numCol;
  int yEnd = yBegin + numCol;
  int y, x, k, outIdx;
  float tmp, s = 0.0;
  
  for(y=yBegin,x=xBegin;y<yEnd;y+=BlockSize,x+=BlockSize){
    Ys[ty][tx] = input[y + ty*numCol + tx];
    Xs[tx][ty] = input[x + ty*numCol + tx];
    __syncthreads();
    for(k=0;k<BlockSize;k++){
      tmp = Ys[ty][k] - Xs[k][tx];
      s += tmp*tmp;
    }
    __syncthreads();
  }
  outIdx = (by*BlockSize + ty) * numRow + bx*BlockSize + tx;
  output[outIdx] = sqrtf(s);
}


__global__ void gpuEuclidean2(const double *inputA,const double *inputB, int numRowA, int numRowB, int numCol, double *output){
  __shared__ float Ys[BlockSize][BlockSize];
  __shared__ float Xs[BlockSize][BlockSize]; 
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int yBegin = by * BlockSize * numCol;
  int xBegin = bx * BlockSize * numCol;
  int yEnd = yBegin + numCol;
  int y, x, k, outIdx;  
  float tmp, s = 0.0;
  
  for(y=yBegin,x=xBegin;y<=yEnd;y+=BlockSize,x+=BlockSize){
    Ys[ty][tx] = inputA[y + ty*numCol + tx];
    Xs[tx][ty] = inputB[x + ty*numCol + tx];
    __syncthreads();
#pragma unroll
    for(k=0;k<BlockSize;k++){
      tmp = Ys[ty][k] - Xs[k][tx];
      s += tmp*tmp;
    }
    __syncthreads();
  }
  outIdx = (by*BlockSize + ty) * numRowB + bx*BlockSize + tx;
  //output[outIdx] = sqrtf(s);
  output[outIdx] = 10;
}
