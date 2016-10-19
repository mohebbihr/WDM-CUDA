function [testpred acc] = knn( traindata,testdata,traingnd,testgnd,k,f)
%traindata nSample * nDim
%testdata nSample * nDim
%traingnd nSample * 1
Distmat = Dist(testdata, traindata);

[~, idx] = sort(Distmat, 2);

if nargin < 5
    k = 1;
end
if nargin < 6
    f = 0;
end

if f == 1
    idx(:,1) = [];
end

%k=1;
%f=0;
testpred = traingnd(idx(:, k));
if nargout > 1
	%acc = sum(testpred(:, 1) == testgnd) / length(testgnd);
    count =0;
    parfor i=1:length(testpred)
        if testpred(i,1)==testgnd(i,1)
            count = count +1;
        end
    end
    acc = count/length(testgnd);
end

end

