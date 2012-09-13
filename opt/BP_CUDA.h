/*********************************************************************************
 * File: BP_CUDA.h
 * Description: Belief Propagation for Ising Model, CUDA parallelized version,
 *              header file for BP_CUDA.cu
 * by BreakDS, @ University of Wisconsin-Madison, Fri Aug 31 15:25:19 CDT 2012
 *********************************************************************************/

#pragma once

#include <cstdio>

namespace optimize_cuda
{
  struct Options
  {
    int maxIter;
    int verbose;
    float lambda;
  };


  
  double LoopyBP( const float *D, // The distance term for each pixel, h x w
                  const float *label,
                  // the label vectors for each candidates of each pixel, h x w x K x dim
                  const int height,
                  const int width,
                  const int K,
                  const int dim,
                  int *result,
                  Options options,
                  float* msgBuf = NULL );
};
