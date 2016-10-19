% The test script for WDM metric.
% author: Hamidreza Mohebbi
% Email: mohebbi.h@gmail.com
% May, 2016
clear all;

disp('Madelon dataset');
trainfea = importdata('madelon/madelon_train.data');
traingnd = importdata('madelon/madelon_train.labels');
testfea = importdata('madelon/madelon_valid.data');
testgnd = importdata('madelon/madelon_valid.labels');
numclass = 2;
% reduce number of features to 10
[~,trainfea] = pcares(trainfea,10);
[~,testfea] = pcares(testfea,10);
traingnd(traingnd == -1) = 2;
testgnd(testgnd == -1) = 2;

options.k1 = 2;
options.k2 = 4; 
   
fea_Train = trainfea(1:10,:);
gnd_Train = traingnd(1:10,:);
fea_Test = testfea(1:5,:);        
gnd_Test = testgnd(1:5,:); 
	
[wpredlabel, wacc, wUProj] = wdm(fea_Train, gnd_Train, fea_Test, gnd_Test,options);

disp(['Total acc:' num2str(wacc * 100)]);




