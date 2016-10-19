function [ acc predictgnd] = evaluate_dimesion( trainfea,traingnd,testfea,testgnd,W,k,f)
%fea, n*d
%gnd, n*1
% W, d*d'
if nargin<6
    k = 1;f = 0;
end
[d dp] = size(W);
acc = zeros(dp,1);
parfor i = 1:dp
    tmptrain= trainfea*W(:,1:i);
    tmptest= testfea*W(:,1:i);
    [ predictgnd(:,i) acc(i)] = knn(tmptrain,tmptest,traingnd,testgnd,k,f);
end



end

