% WDM learning weighted distance metric. 
% author: Hamidreza Mohebbi
% Email: mohebbi.h@gmail.com
% May, 2016

function [ predlabel, acc, UProj ] = wdm( trainfea, traingnd, testfea, testgnd,options )
        		
        [ Udla1, ~, ~] = WDLAMatrix( trainfea,traingnd,options );        
	%[ Udla1, ~, ~] = WDLAMatrixCUDA( trainfea,traingnd,options ); % CUDA call
	acc1 = evaluate_dimesion( trainfea,traingnd, trainfea,traingnd,Udla1, 1, 1);        
        
        [~, d] = max(acc1); 
        testfeaProj = testfea*Udla1(:,1:d);        
        
        [ Udla2, ~, ~] = WDLAMatrix( testfeaProj,testgnd,options );        
        %%[ Udla2, ~, ~] = WDLAMatrixCUDA( testfeaProj,testgnd,options ); % CUDA call
	acc2 = evaluate_dimesion( trainfea,traingnd,trainfea,traingnd,Udla1(:,1:d)*Udla2,1, 1);        
        
        [~, d2] = max(acc2);
        [~, w2] = size(Udla2);
        if w2 < d2
            d2 = w2;        
        end
        UProj = Udla1(:,1:d)*Udla2(:,1:d2);        
        
        trainProj = trainfea*UProj;
        testProj = testfeaProj*Udla2(:,1:d2);        
        
        [ predlabel acc] = knn( trainProj,testProj,traingnd,testgnd); % change this if you want to use another classifier                     
                        
end

