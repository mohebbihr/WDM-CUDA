function [ Udla eign L] = WDLAMatrix( fea,gnd,options )
% Weighted Bipart

% example
% fea = rand(500,100);
% gnd = [ones(250,1);-ones(250,1)];
% options.k1 = 2;
% options.k2 = 3;
% [ Udla] = DLAMatrix( fea,gnd,options );
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

k2 = 3;
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

BlockSize = 16;
% caculate L matrix
L = zeros(sampleNumber,sampleNumber);
%{
idxsame = zeros(1, sampleNumber);
idxdiff = zeros(1, sampleNumber);
samemat = zeros(1, sampleNumber);
diffmat = zeros(1, sampleNumber);
weight = zeros(1, k1 + k2 + 2);
per_idx = zeros(1, k1 + k2 + 1);
omega = zeros(1, k1 + k2 );
WLi = zeros(k1 + k2 + 1);
sidx = zeros(1, sampleNumber);
didx = zeros(1, sampleNumber);
sameid = zeros(1, sampleNumber);
diffid = zeros(1, sampleNumber);

Lcal = parallel.gpu.CUDAKernel('CUDA/gpuWDLA.ptx','CUDA/gpuWDLA.cu','gpuWDLA2');    
Lcal.ThreadBlockSize = [BlockSize BlockSize];
Lcal.GridSize = [ceil(sampleNumber/BlockSize) ceil(sampleNumber/BlockSize)];

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

[LG] = feval(Lcal,DistG,gndG, idxsameG, idxdiffG, samematG, diffmatG, weightG, per_idxG, WLiG, sidxG, didxG, sameidG, diffidG,  LG, omegaG, beta,sampleNumber, k1, k2);
%[LG] = feval(Lcal,LG,k1, k2);
L = gather(LG);

disp('L after GPU');
L(1:5)

disp('size of L');
size(L)
%}

for LoopI = 1:sampleNumber 
    try    
    idxsame = find(gnd == gnd(LoopI));
    idxsame(idxsame == LoopI) = [];
    idxdiff = find(gnd ~= gnd(LoopI));
    if ~isempty(idxsame)
    samemat = Distant(LoopI,idxsame');
    [samedist, sidx] = sort(samemat);
    sameid = idxsame(sidx);
    else
        sameid = zeros(1,100)';
    end
    if ~isempty(idxdiff)
    diffmat = Distant(LoopI,idxdiff');
    [diffdist, didx] = sort(diffmat);
    diffid = idxdiff(didx);
    else
        diffid = zeros(1,100)';
    end
    
    sameclass = sameid(1:k1);
    diffclass = diffid(1:k2);
    % we compute weight for k+1 elements, because we want to avoid weight
    % zero for the last element             
    samedist = samedist(1:k1+1); 
    diffdist = diffdist(1:k2+1);
    % weighted distance     
    weight = [Weight(samedist,k1+1),Weight(diffdist,k2+1)];   
    WLi = WeightLi(weight,k1,k2,beta);
    
    %per_idx : indexes of L's permutation 
    per_idx = [LoopI, sameclass', diffclass'];    
    L(per_idx,per_idx) = L(per_idx,per_idx) + WLi;
    
    catch
        % error;
    end
      
end

%disp('idxsame from CPU');
%idxsame
%disp('idxdiff from CPU');
%idxdiff

disp('L after CPU');
L(1:5)

%disp('size of L');
%size(L)

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

function [Li] = WeightLi(Weight,k1,k2,beta)
% this function computes the weighted Li matrix, according to the Weight
% parameter

omega = ones(1,k1+k2);
omega(:,k1+1:k1+k2) = -beta;
omega = omega.* Weight; 
omega = omega';
sumomega = sum(omega);
Li = [sumomega,-omega';-omega,diag(omega)];

end

function [W] = Weight(vector,k)
% We compute weight for a vector. We assumed that the k is always greater
% than one
k1 = floor(k/2);
k2 = k-k1;
W1 = ones(1,k1);

vector = vector(k1+1:end);

t1 = vector(k2) - vector(1);
t2 = vector(k2) + vector(1);
c1 = vector(k2) - vector;
c2 = vector(k2) + vector;

W2 = (c1/t1).* (t2./c2);

W2(k2)=[];

W=[W1,W2];

end

