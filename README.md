# WDM-CUDA

This repository contains the source codes of the new distance metric named WDM and its parallel GPU (CUDA) implementation. The aim of this project is to builds a weighted-distance metric from the labeled training set and combines it with group-level information from the
unlabeled test set.  This differs from the related approaches that use only labeled training data in order to build the model.  The WDM is a
semi-supervised approach, similar to Semi-supervised Discriminant Analysis and Bipart that used labeled and unlabeled data.  The difference between WDM and methods
like SDA is that it learns group-level information from both training and test sets.  The proposed method combines the information from these
two sources with a projection matrix.  In addition, similar and dissimilar samples contribute to this process based on their similarity and
dissimilarity to the query sample.  In order to reach this goal, WDM assigns a weight to each sample. You can read about this new method on 
a paper entitled "Learning Weighted Distance Metric From Group Level Information and Its Parallel Implementation" on this [link](http://link.springer.com/article/10.1007/s10489-016-0826-7).
If you want to use this code, please cite this paper.

The Matlab implementation contains CUDA implementation too that for big data shows its power and reduce the execution time. The CUDA folder contains
gpuWDLA.cu file that is the CUDA implementation of WDM method. The WDLAMatrixCUDA.m file use CUDA implementation. The CPU implementation of this method use WDLAMatrix.m file.

In order to run this code, you just need to run test.m file. This file loads madelon data set and computes classification using KNN method. You can
use WDM metric with other classifications too. You need to replace the KNN with other classifier in the WDM.m file. 


