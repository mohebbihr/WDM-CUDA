#include "min_reduction.cu"

#define max(a,b) ((a) > (b) ? (a) : (b))

// samemat2 => vector = vector(k1+1:end);
__device__ void ComputeWeight(double * weight, double * samemat, double * diffmat, int len, int k1, int k2, int loopIdx, int numRow)
{
        int i;
        double s_t1, s_t2;
	double d_t1, d_t2;

        for(i=0; i<k1/2; i++) weight[loopIdx * len  + i] = 1;
	for(i= (k1 - 1); i< (k1 + k2/2 - 1); i++) weight[loopIdx * len + i] = 1;

        s_t1 = samemat[loopIdx * numRow + k1 - 1] - samemat[loopIdx * numRow + k1/2];
        s_t2 = samemat[loopIdx * numRow + k1 - 1] + samemat[loopIdx * numRow + k1/2];
	d_t1 = diffmat[loopIdx * numRow + k2 - 1] - diffmat[loopIdx * numRow + k2/2];
	d_t2 = diffmat[loopIdx * numRow + k2 - 1] + diffmat[loopIdx * numRow + k2/2];

        for(i=k1/2; i<(k1 -1); i++){
		weight[loopIdx * len + i] = ((samemat[loopIdx * numRow + k1 - 1] - samemat[loopIdx * numRow + i]) / s_t1 ) * (s_t2 / (samemat[loopIdx * numRow + k1 - 1] + samemat[loopIdx * numRow + i]));
        }
	for(i= (k1 + k2/2 - 1) ; i< (k1 + k2 - 1); i++){
		weight[loopIdx * len + i] = ((diffmat[loopIdx * numRow + k2 - 1] - diffmat[loopIdx * numRow + (i - k1 + 1)]) / d_t1 ) * (d_t2 / (diffmat[loopIdx * numRow + k2 - 1 ] + diffmat[loopIdx * numRow + (i - k1 + 1)]));
	}

}

__device__ void WeightLi(double * WLi, double * weight, double * omega, int k1, int k2, double beta, int loopIdx, int numRow, int wli_w, int w_w, int o_w )
{
        int i,j;
        double sumomega = 0.0;

        for(i=0; i< k1; i++) omega[loopIdx * o_w + i] = 1;
        for(i=k1; i< (k1 + k2); i++) omega[loopIdx * o_w + i] = - beta;
        for(i=0; i< (k1 + k2); i++) omega[loopIdx * o_w + i] = omega[loopIdx * o_w + i] * weight[loopIdx * w_w + i];
        for(i=0; i< (k1 + k2); i++) sumomega += omega[loopIdx * o_w + i];
        // create WLi matrix
        //WLi = [sumomega,-omega';-omega,diag(omega)];
        // create diag
        for(i=1; i< (k1 + k2 + 1); i++)
        	for(j=1; j< (k1 + k2 + 1); j++){
                        if(i==j) WLi[loopIdx * wli_w + i * (k1 + k2 + 1) + j] = omega[loopIdx * o_w + i-1];
                        else    WLi[loopIdx * wli_w + i * (k1 + k2 + 1) + j] = 0.0;
                }

        for(i=1; i< (k1 + k2 + 1); i++){
                WLi[loopIdx * wli_w + i] = -omega[loopIdx * o_w + i-1]; // WLi[0][i]
                WLi[loopIdx * wli_w + i * wli_w] = -omega[loopIdx * o_w + i-1]; // WLi[i][0]
        }
	
        WLi[loopIdx * wli_w * wli_w] = sumomega;

}

__device__ void selection_sort(double *data, double *index, int left, int right, int loopIdx, int numRow, int k)
{
    for (int i = left ; i <= right ; ++i)
    {
        double min_val = data[loopIdx * numRow + i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            double val_j = data[loopIdx * numRow + j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[loopIdx * numRow + min_idx] = data[loopIdx * numRow + i];
            data[loopIdx * numRow + i] = min_val;
            // swap index values
            double tmp = index[loopIdx * k + i];
            index[loopIdx * k + i] = index[loopIdx * k + min_idx];
            index[loopIdx * k + min_idx] = tmp;
        }
    }
}


// L is the output 

__global__ void gpu_WDLA(double * out, double * per_idx,double * WLi, double * omega, double beta, double * weight, double * sameid, double * diffid, double * samemat, double * diffmat,double * sidx, double * didx, double * dist, double * idxsame, double * idxdiff, double * gnd, int numRow, int numElemPerThread, int k1, int k2){

  int thread_start_idx = (threadIdx.x + blockIdx.x * blockDim.x) * numElemPerThread;
  int thread_end_idx = thread_start_idx + numElemPerThread;

  // int y, x, k, outIdx;
  int gndIdx, i, j, loopIdx, myclass;
  // double tmp, s = 0.0;
  int idxsame_i =0, idxdiff_i =0;

  for(loopIdx = thread_start_idx; loopIdx< thread_end_idx && loopIdx < numRow; loopIdx++){

  for(gndIdx =0; gndIdx < numRow; gndIdx ++){
	myclass = (int) gnd[loopIdx];
        if(myclass == gnd[gndIdx] && gndIdx != loopIdx ){
		idxsame[loopIdx * numRow + idxsame_i] = gndIdx;
                idxsame_i ++;
        }
        if(gnd[gndIdx] != myclass){
                idxdiff[loopIdx * numRow + idxdiff_i] = gndIdx;
                idxdiff_i ++;
        }
  }
  
  // samemat = Distant(LoopI,idxsame');
  // diffmat = Distant(LoopI,idxdiff');
  if(idxsame_i > 0){
        for(i=0; i< idxsame_i; i++){
		samemat[loopIdx * numRow + i] = dist[loopIdx * numRow + (int)idxsame[loopIdx * numRow + i]];
        }
  }
  if(idxdiff_i > 0){
        for(i=0; i< idxdiff_i; i++){
                diffmat[loopIdx * numRow + i] = dist[loopIdx * numRow + (int)idxdiff[loopIdx * numRow + i]];
        }
  }
  
  // [samedist, sidx] = sort(samemat);
  for(i=0; i< max(idxsame_i, idxdiff_i); i++){
        sidx[loopIdx * (k1 + 1) + i] = i;
        didx[loopIdx * (k2 + 1) + i] = i;
  }

  /*for(i=0; i< k1; i++){
	cudaStream_t s1;
    	cudaStreamCreateWithFlags( &s1, cudaStreamNonBlocking );
    	min_reduce<<<NumBlock,NumThread>>>(samemat,samemat, loopIdx, idxsame_i ); 
    	cudaStreamDestroy(s1);
	
  }*/
  selection_sort(samemat, sidx, 0, idxsame_i -1, loopIdx, numRow, k1);
  selection_sort(diffmat, didx, 0, idxdiff_i -1, loopIdx, numRow, k2);

  // sameid = idxsame(sidx);
  for(i=0; i< idxsame_i; i++){
        sameid[loopIdx * (k1 + 1) + i] = idxsame[loopIdx * numRow + (int)sidx[loopIdx * (k1 + 1) + i]];
  }
  for(i=0; i< idxdiff_i; i++){
        diffid[loopIdx * (k2 + 1) + i] = idxdiff[loopIdx * numRow + (int)didx[loopIdx * (k2 + 1) + i]];
  }
  
  // weighted distance
  ComputeWeight(weight, samemat, diffmat, k1 + k2 + 2, k1 + 1, k2 + 1, loopIdx, numRow);
  WeightLi(WLi, weight, omega, (int)k1, (int)k2, beta, loopIdx, numRow, k1 + k2 + 1, k1 + k2 + 2, k1 + k2);

  // per_idx = [LoopI, sameclass', diffclass'];
  per_idx[loopIdx * numRow] = loopIdx;
  for(i=1; i<= k1; i++) per_idx[loopIdx * (k1 + k2 + 1) + i] = sameid[loopIdx * (k1 + 1) + i-1];
  for(i= (k1 + 1); i< (k1 + k2 + 1); i++) per_idx[loopIdx * (k1 + k2 + 1) + i] = diffid[loopIdx * (k2 + 1) + i - (int)(k1 + 1)];

  __syncthreads();
  // L(per_idx,per_idx) = L(per_idx,per_idx) + WLi;
  for(i=0; i< (k1 + k2 + 1); i++){
        for(j=0; j< (k1 + k2 + 1); j++){
                out[ (int)per_idx[i] * numRow + (int)per_idx[j]] = out[ (int)per_idx[i] * numRow + (int)per_idx[j]] + WLi[loopIdx * (k1 + k2 + 1) * (k1 + k2 + 1) + i * (k1 + k2 + 1) + j ];
                __syncthreads();
        }
        __syncthreads();
  }

  __syncthreads();

  } // end of loopIdx for
}

