for i= 2.^[0:4]
%i = 8;
        disp(['Number of training records' num2str(1000 * i)]);
	% class labels: 1,2,3,4 for training set
	trainfea = rand(i * 1000,400); %#dimension * #features
	% class labels: 1,2,3,4 for training set
	traingnd = [ones((i * 1000)/4,1) * 1; ones((i * 1000)/4,1) * 2; ones((i * 1000)/4,1) * 3; ones((i * 1000)/4,1) * 4];

	testfea = rand(1000,400);
	% groups: 2, 3, 4, 5 for test set
	testgroup = [ones(250,1) * 2; ones(250,1) * 3; ones(250,1) * 4; ones(250,1) * 5];
	options.k1 = 2;
	options.k2 = 4;

	tic();
	[predlabel, acc, UProj] = wdm(trainfea, traingnd, testfea, testgroup, options);
	total = toc();

	disp(['WDM with CUDA - Total time: ' num2str(total)]);
end
