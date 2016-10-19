function [ predlabel, acc, UProj ] = wbipart( trainfea, traingnd, testfea, testgnd,options )		       
        		
        [ Udla1 , ~, ~] = WDLAMatrix( trainfea,traingnd,options );        
		acc1 = evaluate_dimesion( trainfea,traingnd, trainfea,traingnd,Udla1, 1, 1);        
        
        [~, d] = max(acc1); 
        testfeaProj = testfea*Udla1(:,1:d);        
        
        [ Udla2 , ~, ~] = WDLAMatrix( testfeaProj,testgnd,options );        
        acc2 = evaluate_dimesion( trainfea,traingnd,trainfea,traingnd,Udla1(:,1:d)*Udla2,1, 1);        
        
        [~, d2] = max(acc2);
        [~, w2] = size(Udla2);
        if w2 < d2
            d2 = w2;        
        end
        UProj = Udla1(:,1:d)*Udla2(:,1:d2);        
        
        trainProj = trainfea*UProj;
        testProj = testfeaProj*Udla2(:,1:d2);        
        
        [ predlabel, acc] = knn( trainProj,testProj,traingnd,testgnd);                      
                        
end

