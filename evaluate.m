function [ precision, recall, accuracy] = evaluate( pregnd,gnd,posvalue)
%%%%%%%%
%pregnd: predict value
%gnd: ground truth
%posvalue: positive value in gnd. For example: gnd contains -1,1, if 1 is
%the positive value, then posvalue = 1;

idx = (gnd()==posvalue);

p = length(gnd(idx));
n = length(gnd(~idx));
N = p+n;

tp = sum(gnd(idx)==pregnd(idx));
tn = sum(gnd(~idx)==pregnd(~idx));
fp = n-tn;

tp_rate = tp/p;

accuracy = (tp+tn)/N;
precision = tp/(tp+fp);
recall = tp_rate;


%{
num = length(pregnd);
tp = 0;
fp = 0;
fn = 0;
tn = 0;
for i = 1:num
    if pregnd(i) == gnd(i)
        if pregnd(i) == posvalue
            tp = tp + 1;
        else
            tn = tn + 1;
        end
    else
        if pregnd(i) == posvalue
            fp = fp + 1;
        else
            fn = fn + 1;
        end
    end
end

precision = tp/(tp+fp);
recall = tp/(tp+fn);
f1 = 2*precision*recall/(precision+recall);
accuracy = (tp+tn)/num;
%}
end

