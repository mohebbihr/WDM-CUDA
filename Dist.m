function Distmat = Dist(A, B,Lnorm)
% kevin dist
%A  m*p
%B  n*p

% Distmat m*n

% look at the pdist2 function in matlab help

if nargin <= 2
    Lnorm = 2;
end
if Lnorm == 1
    Distmat = A*ones(size(B'))-ones(size(A))*(B');
elseif Lnorm ==2    
    Distmat = A.^2*ones(size(B'))+ones(size(A))*(B').^2-2*A*B';
end

%{
% it is L-infinity norm
[r,c] = size(Distmat);
for i=1:r
    for j=1:c
        if Distmat(i,j) ~= norm(Distmat(i,:),'inf')
            Distmat(i,j)=0;
        end
    end
end
%}
%Distmat = pdist2(A,B);

end

