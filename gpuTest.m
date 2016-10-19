gpuDevice(1);

A = rand(16, 16);
B = rand(16, 16);

[m,p] = size(A);
[n,p] = size(B);

BlockSize = 16;

mRem = BlockSize - rem(m,BlockSize);
nRem = BlockSize - rem(n,BlockSize);
mPad = m + mRem;
nPad = n + nRem;

pPad = 2;
while (pPad < p)
  pPad = pPad * 2;
end
pRem = pPad - p;
APad = padarray(A,[mRem pRem]);
BPad = padarray(B,[nRem pRem]);

Euclidean = parallel.gpu.CUDAKernel('CUDA/gpuEuclidean.ptx','CUDA/gpuEuclidean.cu','gpuEuclidean2');

    Euclidean.ThreadBlockSize = [BlockSize BlockSize];
    Euclidean.GridSize = [mPad/BlockSize nPad/BlockSize];

    AG = gpuArray(APad);
    BG = gpuArray(BPad);
    resultG = gpuArray(zeros(m,n));

[resultG] = feval(Euclidean,AG,BG,m,n,p,resultG);
Distmat = gather(resultG);

disp('GPU result');
Distmat(1:5)



