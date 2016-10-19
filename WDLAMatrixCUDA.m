% The CUDA implementation of WDLAMatrix. This script calls CUDA functions to perfom the calculations.
% author: Hamidreza Mohebbi
% Email: mohebbi.h@gmail.com
% May, 2016

function [ Udla eign L] = WDLAMatrixCUDA( fea,gnd,options )

% example
% fea = rand(500,100);
% gnd = [ones(250,1);-ones(250,1)];
% options.k1 = 2;
% options.k2 = 3;
% [ Udla] = WDLAMatrixCUDA( fea,gnd,options );
% d = 50;
% proj = fea*Udla(:,1:d);

k1 = 2;
if isfield(options,'k1')
    k1 = options.k1;
end

m = zeros(size(fea,1),1)+1;
if isfield(options,'m')
    m = options.m;
end

k2 = 4;
if isfield(options,'k2')
    k2 = options.k2;
end

beta = 0.3;
if isfield(options,'beta')
    beta = options.beta;
end

[sampleNumber, numCol] = size(fea);
X = fea';
% build patch
Distant = Dist(fea,fea);

NumBlock = 32;
NumThread = 512;

NumElemPerThread = ceil(sampleNumber/(NumBlock * NumThread));

% caculate L matrix
L = zeros(sampleNumber, sampleNumber);
idxsame = zeros(sampleNumber, sampleNumber);
idxdiff = zeros(sampleNumber, sampleNumber);
samemat = zeros(sampleNumber, sampleNumber);
diffmat = zeros(sampleNumber, sampleNumber);
weight = zeros(sampleNumber, k1 + k2 + 2);
per_idx = zeros(sampleNumber, k1 + k2 + 1);
omega = zeros(sampleNumber, k1 + k2 );
WLi = zeros(k1 + k2 + 1, k1 + k2 + 1, sampleNumber);
sidx = zeros(sampleNumber, k1 + 1);
didx = zeros(sampleNumber, k2 + 1);
sameid = zeros(sampleNumber, k1 + 1);
diffid = zeros(sampleNumber, k2 + 1);

gpudev = gpuDevice(1);

Lcal = parallel.gpu.CUDAKernel('CUDA/gpuWDLA.ptx','CUDA/gpuWDLA.cu','gpu_WDLA');    

Lcal.ThreadBlockSize = [NumThread 1];
Lcal.GridSize = [NumBlock 1];

gndG = gpuArray(gnd);
DistG = gpuArray(Distant);
idxsameG = gpuArray(idxsame);
idxdiffG = gpuArray(idxdiff);
samematG = gpuArray(samemat);
diffmatG = gpuArray(diffmat);
weightG = gpuArray(weight);
per_idxG = gpuArray(per_idx);
WLiG = gpuArray(WLi);
sidxG = gpuArray(sidx);
didxG = gpuArray(didx);
sameidG = gpuArray(sameid);
diffidG = gpuArray(diffid);
omegaG = gpuArray(omega);
LG = gpuArray(L);    

[LG] = feval(Lcal,LG, per_idxG, WLiG, omegaG, beta, weightG, sameidG, diffidG, samematG, diffmatG, sidxG, didxG, DistG, idxsameG,idxdiffG, gndG, sampleNumber, NumElemPerThread, k1, k2);

L = gather(LG);
L = L';      

%mat = X * L * X';
mat = X * L;
mat(isnan(mat)) = 0;
mat = mat * X';
mat(isnan(mat)) = 0;
mat = (mat + mat')/2;
[V,D] = eig(mat);
eign = diag(D) ;
[eign, index] = sort(eign);
Udla = V(:, index(1:end));


end

