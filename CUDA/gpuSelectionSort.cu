/*
* Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#include <iostream>
#include <cstdio>
#include <helper_cuda.h>
#include <helper_string.h>

#define MAX_DEPTH       16
#define INSERTION_SORT  32

////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__global__ void selection_sort(double *data, double *index, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        float min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            float val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
			// swap index values
			double tmp = index[i];
			index[i] = index[min_idx];
			index[min_idx] = tmp;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
/*
__global__ void gpuQSort(double *data, double *index, int left, int right, int depth)
{
    // If we're too deep or there are few elements left, we use an insertion sort...
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT)
    {
        selection_sort(data,index, left, right);
        return;
    }

    double *lptr = data+left;
    double *rptr = data+right;
    double  pivot = data[(left+right)/2];

    double *lptr_index = index+left;
    double *rptr_index = index+right;
	
    // Do the partitioning.
    while (lptr <= rptr)
    {
        // Find the next left- and right-hand values to swap
        double lval = *lptr;
        double rval = *rptr;

	double lval_index = *lptr_index;
        double rval_index = *rptr_index;
		
        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot)
        {
            lptr++;
            lval = *lptr;
	    // index array
	    lptr_index ++;
	    lval_index = *lptr_index;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot)
        {
            rptr--;
            rval = *rptr;
	    // index array
	    rptr_index --;
	    rval_index = *rptr_index;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr)
        {
            *lptr++ = rval;
            *rptr-- = lval;
	    // index array
	    *lptr_index++ = rval_index;
            *rptr_index-- = lval_index;			
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data))
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        gpuQSort<<< 1, 1, 0, s >>>(data, index, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right)
    {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        gpuQSort<<< 1, 1, 0, s1 >>>(data, index, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}
*/
