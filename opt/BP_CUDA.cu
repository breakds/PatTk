/*********************************************************************************
 * File: BP_CUDA.cu
 * Description: Belief Propagation for Ising Model, CUDA parallelized version
 * by BreakDS, @ University of Wisconsin-Madison, Fri Aug 31 15:25:19 CDT 2012
 *********************************************************************************/

#include "BP_CUDA.h"
#include <cuda.h>
#include "../utils/cuda_aux.hpp"





int host_patch_side;

  

namespace optimize_cuda
{

  __constant__ int g_inc[4];
  __constant__ float g_coeff[6];
  __constant__ int g_patch_side;

  
  // This function calculate the distance between vector
  // ( a0, ..., a5 ) and ( b0, ..., b5 )
  // direction should be specified

  // __device__ inline float dist_device( const float a0, const float a1,
  //                                      const float a2, const float a3,
  //                                      const float a4, const float a5,
  //                                      const float b0, const float b1,
  //                                      const float b2, const float b3,
  //                                      const float b4, const float b5,
  //                                      int direction )
  // {
  //   float tmp;

  //   // [0] = image index
  //   tmp = fabsf( b0 - a0 );
  //   if ( tmp > 0.5 ) return 150000.0;

  //   // [1], [2] = dy, dx (rotation representation)
  //   tmp = fabsf( a1 - b1 ) + fabsf( a2 - b2  );
  //   if ( tmp > 1.0 ) {
  //     return 150000.0;
  //   }

      
  //   // [4],[5] = spatial distance
  //   // Should be compensated by the rotation (dy,dx)
  //   // Note that dy = b1 dx = b2 by definition

  //   float ay(a4), ax(a5);
    
  //   if ( 0 == direction ) {
  //     ay -= b2;
  //     ax -= b1;
  //   } else if ( 1 == direction ) {
  //     ay += b1;
  //     ax -= b2;
  //   } else if ( 2 == direction ) {
  //     ay += b2;
  //     ax += b1;
  //   } else if ( 3 == direction ) {
  //     ay -= b1;
  //     ax += b2;
  //   }
    
    
    
      
  //   tmp = fabsf( ay - b4 ) + fabsf( ax - b5 );
  //   if ( tmp > g_patch_side ) {
  //     return 150000.0;
  //   }
      
  //   // [1],[2] = spatial distance
    
  //   float sum = tmp * g_coeff[4];

  //   sum += fabsf( b1 - a1 ) * g_coeff[1];
  //   sum += fabsf( b2 - a2 ) * g_coeff[2];
  //   sum += fabsf( b3 - a3 ) * g_coeff[3];

    
  //   return sum;
  // }

  // // The host version of dist_device()
  // inline float dist_host( const float a0, const float a1,
  //                         const float a2, const float a3,
  //                         const float a4, const float a5,
  //                         const float b0, const float b1,
  //                         const float b2, const float b3,
  //                         const float b4, const float b5,
  //                         int direction )
  // {
  //   const float coeff[6] = { 0.0, 30.0, 30.0, 10.0, 1.0, 1.0 };
  //   float tmp;

  //   // [0] = image index
  //   tmp = fabsf( b0 - a0 );
  //   if ( tmp > 1.0 ) return 150000.0;

  //   // [1], [2] = dy, dx (rotation representation)
  //   tmp = fabsf( a1 - b1 ) + fabsf( a2 - b2  );
  //   if ( tmp > 1.0 ) {
  //     return 150000.0;
  //   }

      
  //   // [4],[5] = spatial distance
  //   // Should be compensated by the rotation (dy,dx)
  //   // Note that dy = b1 dx = b2 by definition

  //   float ay(a4), ax(a5);

  //   if ( 0 == direction ) {
  //     ay -= b2;
  //     ax -= b1;
  //   } else if ( 1 == direction ) {
  //     ay += b1;
  //     ax -= b2;
  //   } else if ( 2 == direction ) {
  //     ay += b2;
  //     ax += b1;
  //   } else if ( 3 == direction ) {
  //     ay -= b1;
  //     ax += b2;
  //   }
    
    
    
      
  //   tmp = fabsf( ay - b4 ) + fabsf( ax - b5 );
  //   if ( tmp > host_patch_side ) {
  //     return 150000.0;
  //   }
      
  //   // [1],[2] = spatial distance
    
  //   float sum = tmp * coeff[4];

  //   sum += fabsf( b1 - a1 ) * coeff[1];
  //   sum += fabsf( b2 - a2 ) * coeff[2];
  //   sum += fabsf( b3 - a3 ) * coeff[3];

    
  //   return sum;
  // }



  __device__ inline float dist_device( const float a0, const float a1,
                                       const float a2, const float a3,
                                       const float a4, const float a5,
                                       const float b0, const float b1,
                                       const float b2, const float b3,
                                       const float b4, const float b5 )
  {
    float tmp;

    // [0] = image index
    tmp = fabsf( b0 - a0 );
    if ( tmp > 0.5 ) return 150000.0;

    tmp = fabsf( a1 - b1 ) + fabsf( a2 - b2  );
    if ( tmp > 1.5 ) {
      return 150000.0;
    }

    tmp = 0.0f;
    
    tmp += fabsf( b1 - a1 ) * g_coeff[1];
    tmp += fabsf( b2 - a2 ) * g_coeff[2];
    tmp += fabsf( b4 - a4 ) * g_coeff[4];
    tmp += fabsf( b5 - a5 ) * g_coeff[5];

    return tmp;
  }

  // The host version of dist_device()
  inline float dist_host( const float a0, const float a1,
                          const float a2, const float a3,
                          const float a4, const float a5,
                          const float b0, const float b1,
                          const float b2, const float b3,
                          const float b4, const float b5 )
  {
    const float coeff[6] = { 0.0, 30.0, 30.0, 10.0, 1.0, 1.0 };
    float tmp;

    // [0] = image index
    tmp = fabsf( b0 - a0 );
    if ( tmp > 1.0 ) return 150000.0;

    tmp = fabsf( a1 - b1 ) + fabsf( a2 - b2  );
    if ( tmp > 1.5 ) {
      return 150000.0;
    }

    tmp = 0.0f;
    
    tmp += fabsf( b1 - a1 ) * coeff[1];
    tmp += fabsf( b2 - a2 ) * coeff[2];
    tmp += fabsf( b4 - a4 ) * coeff[4];
    tmp += fabsf( b5 - b5 ) * coeff[5];

    return tmp;
  }

  

  // normalize messages of a node (pixel) with the same direction (sum to 0)
  // agentNum: 4 * #pixel (4 = number of directions)
  // msg: the message array [dir(4), pixel(#pixel), k(K)]
  // K: # of candidate labels
  __global__ void NormalizeMsg_device( const int agentNum, 
                                       float *msg,
                                       int K )
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      float offset = 0.0f;
      for ( int k=0; k<K; k++ ) {
        offset += msg[ idx * K + k ];
      }
      offset /= K;
      for ( int k=0; k<K; k++ ) {
        msg[ idx * K + k ] -= offset;
      }
    }
  }

  
  // Update the result array (one cadidate label id for each pixel)
  // agentNum: # of pixel
  // D: self distance matrix [pixel(# of pixels), k(K)];
  // msg: the message array [dir(4), pixel(# of pixels), k(K)]
  // result: the result array [pixel(# of pixels)]
  // K: # of labels per pixel
  __global__ void UpdateResult_device( const int agentNum, const float *D,
                                       const float* msg,
                                       int *result, int K )
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if ( idx < agentNum ) {
      result[idx] = 0;
      float min = 0.0f;
      for ( int k=0; k<K; k++ ) {
        float sum = D[idx*K+k];
        for ( int dir=0; dir<4; dir++ ) {
          sum += msg[ ( dir * agentNum + idx ) * K + k ];
        }

        if ( 0 == k ) {
          min = sum;
        } else if ( sum < min ) {
          min = sum;
          result[idx] = k;
        }
      }
    }
  }

  

  double UpdateEnergy( const float *D, const float *label, int height, int width, int K,
                       int dim, float lambda, int *result )
  {

    // constants
    const int inc[4] = { -width, -1, width, 1 };
    const int incDim[4] = {-width*K*dim,-K*dim,width*K*dim,K*dim};
    
    
    double energy = 0.0;
    int i = 0;
    const float *labelp = label;
    for ( int y=0; y<height; y++ ) {
      for ( int x=0; x<width; x++ ) {
        energy += D[i*K+result[i]];
        
        // UP:
        int d = 0;
        if ( y > 0 ) {
          const float *lp0 = labelp + result[i] * dim;
          const float *lp1 = labelp + incDim[d] + result[i+inc[d]] * dim;
          energy += dist_host( lp0[0], lp0[1], lp0[2], lp0[3], lp0[4], lp0[5],
                               lp1[0], lp1[1], lp1[2], lp1[3], lp1[4], lp1[5] ) * lambda;
        }
        

        // LEFT:
        d = 1;
        if ( x > 0 ) {
          const float *lp0 = labelp + result[i] * dim;
          const float *lp1 = labelp + incDim[d] + result[i+inc[d]] * dim;
          energy += dist_host( lp0[0], lp0[1], lp0[2], lp0[3], lp0[4], lp0[5],
                               lp1[0], lp1[1], lp1[2], lp1[3], lp1[4], lp1[5] ) * lambda;
        }

        i++;
        labelp += K * dim;
      }
    }

    return energy;
  }

  
  // Main Kernel Function, which updates messages
  // Currently this function takes 82.3% of the total cuda computation time
  // 
  // agentNum: columns or rows, depnding on the direction
  // D: self distance matrix [pixel(# of pixels), k(K)];
  // msg: the message array [dir(4), pixel(#pixel), k(K)]
  // distance: inter-pixel label distance matrix [k(K), dir(4), pixel(# of pixels)]
  // begins: array [pixel(agentNum)], where begins[i] is the first pixel of agent i
  // ends: array [pixel(agentNum)], where ends[i] is the last (exluded) pixel of agent i
  // K: # of candidate labels
  // lambda: coeffiecient of the inter-pixel label distance term
  // dim: dimension of a label
  // dir: current propagation direction
  // inc: pixel increment in this direction
  // incK: pixel increment, K times as larger as inc
  // area: # of pixels
  __global__ void UpdateMessage_device( int agentNum, // number of agents needes
                                        float *D,
                                        float* msg,
                                        float* distance,
                                        int *begins,
                                        int *ends,
                                        int K,
                                        float lambda,
                                        int dim,
                                        int dir,
                                        int inc,
                                        int incK,
                                        int area )
  {
    
    extern __shared__ float buf[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      int k = threadIdx.y;
      float *h = buf;
      int opp = (dir+2) & 3;
      float *Dp = D + begins[idx] * K;

      int end = ends[idx];
      for ( int i=begins[idx]; i!=end; i+=inc, Dp+=incK ) {
        
        h[ k * blockDim.x + threadIdx.x ] = Dp[k];
        for ( int j=0; j<4; j++ ) {
          if ( j != opp ) {
            h[ k * blockDim.x + threadIdx.x ] += msg[ j * area * K + i * K + k ];
          }
        }
        __syncthreads();

        float min = distance[ ( k * 4 + dir ) * area  + i ] * lambda + h[ threadIdx.x ];


        for ( int k0=1; k0<K; k0++ ) {
          __syncthreads();
          float value = distance[ ( ( k0 * K + k ) * 4 + dir ) * area  + i ] * lambda +
            h[ k0 * blockDim.x + threadIdx.x];
          if ( value < min ) min = value;
        }
        msg[ dir * area * K + ( i + inc ) * K + k ] = min;
        
      } // end for i
    }
  }



  // Precompute the distance matrix that will be heavily used later
  // area: # of pixels, also serve as the agentNum
  // label: the label matrix [pixel(area), k(K), component(dim)]
  // distance: inter-pixel label distance matrix [k(K), dir(4), pixel(# of pixels)]
  // dim: dimension of each label
  // K: # of candidate labels per pixel
  __global__ void Precomputing_agent_float( int area,
                                            float *label,
                                            float *distance,
                                            int dim,
                                            int K )
  {
    int pixel = blockDim.x * blockIdx.x + threadIdx.x;
    if ( pixel < area ) {
      int dir = blockIdx.y;
      // int k0 = threadIdx.y;
      // int k1 = threadIdx.z;
      
      int idx = ( ( threadIdx.y * K + threadIdx.z ) * 4 + dir ) * area + pixel;
      // overflow checking
      if ( pixel + g_inc[dir] < 0 || pixel + g_inc[dir] >= area ) {
        return ;
      }
      
      float *lp0 = label + ( pixel * K + threadIdx.y ) * dim;
      float *lp1 = label + ( ( pixel + g_inc[dir] ) * K + threadIdx.z ) * dim;
      
      distance[idx] = dist_device( lp0[0], lp0[1], lp0[2], lp0[3], lp0[4], lp0[5],
                                   lp1[0], lp1[1], lp1[2], lp1[3], lp1[4], lp1[5] );
    }
  }




  
  
  double LoopyBP( const float *D, 
                  const float *label,
                  const int height,
                  const int width,
                  const int K,
                  const int dim,
                  int *result,
                  Options options, float* msgBuf )
  {

    
    
    const float coeff[6] = { 0.0, 30.0, 30.0, 10.0, 1.0, 1.0 };
    HANDLE_ERROR( cudaMemcpyToSymbol( optimize_cuda::g_coeff, &coeff, sizeof(float) * 6, 0, cudaMemcpyHostToDevice ) );    

    int patchSide = 17;
    HANDLE_ERROR( cudaMemcpyToSymbol( optimize_cuda::g_patch_side, &patchSide, sizeof(int), 0, cudaMemcpyHostToDevice ) );
    host_patch_side = patchSide;

    const int inc[4] = { -width, -1, width, 1 };
    HANDLE_ERROR( cudaMemcpyToSymbol( optimize_cuda::g_inc, inc, sizeof(int) * 4, 0, cudaMemcpyHostToDevice ) );

    const int incK[4] = {-width*K,-K,width*K,K};
    const int order[4] = {2,0,3,1}; // DOWN, UP, RIGHT, LEFT
    const int area = height * width;

    
    // Make sure that the message storage is allocated
    float *buf = msgBuf;
    if ( NULL == buf ) {
      // Message buffer is not provided externally
      buf = new float[4*width*height*K];
    }



    // Initialization of device memory
    // Data term array
    float *devD = NULL;
    HANDLE_ERROR( cudaMalloc( (void**) &devD, sizeof(float) * area * K ) );
    HANDLE_ERROR( cudaMemcpy( devD, D, sizeof(float) * area * K, cudaMemcpyHostToDevice ) );

    // Labels array
    float *devLabel = NULL;
    HANDLE_ERROR( cudaMalloc( (void**) &devLabel, sizeof(float) * area * K * dim ) );
    HANDLE_ERROR( cudaMemcpy( devLabel, label, sizeof(float) * area * K * dim, cudaMemcpyHostToDevice ) );
    
    // Messages
    // After these lines, msg[0] = up, msg[1] = left, msg[2] = right, msg[3] = down
    float *devMsg = NULL;
    HANDLE_ERROR( cudaMalloc( (void**) &devMsg, sizeof(float) * area * K * 4 ) );
    HANDLE_ERROR( cudaMemset( devMsg, 0, sizeof(float) * area * K * 4 ) );
    

    // begins and ends
    int longer = height > width ? height : width;
    int *devBegins = NULL;
    int *devEnds = NULL;
    int begins[longer];
    int ends[longer];
    HANDLE_ERROR( cudaMalloc( (void**) &devBegins, sizeof(int) * longer ) );
    HANDLE_ERROR( cudaMalloc( (void**) &devEnds, sizeof(int) * longer ) );

    // Result
    int *devResult = NULL;
    HANDLE_ERROR( cudaMalloc( (void**) &devResult, sizeof(int) * area * K ) );

    // Intermediate Distance Calculation
    float *devDistance = NULL;
    HANDLE_ERROR( cudaMalloc( (void**) &devDistance, sizeof(float) * K * K * 4 * area ) );

    dim3 precompute_grid( ( area + 1 ) / 8, 4 );
    dim3 precompute_block( 8, K, K );
    


    Precomputing_agent_float<<<precompute_grid, precompute_block>>>( area,
                                                                     devLabel,
                                                                     devDistance,
                                                                     dim,
                                                                     K );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    



    double energy(0);

    for ( int iter=0; iter<options.maxIter; iter++ ) {
      for ( int dirID=0; dirID<4; dirID++ ) {
        int dir = order[dirID];

        // fill in begins and ends
        int agentNum = ( 0 == ( dir & 1 ) ) ? width : height;
        if ( 0 == dir ) {
          // UP:
          for ( int scan=0; scan<agentNum; scan++ ) {
            begins[scan] = scan + width * ( height - 1 );
            ends[scan] = scan;
          }
        } else if ( 1 == dir ) {
          // LEFT:
          for ( int scan=0; scan<agentNum; scan++ ) {
            begins[scan] = width * scan + width - 1;
            ends[scan] = width * scan;
          }
        } else if ( 2 == dir ) {
          // DOWN:
          for ( int scan=0; scan<agentNum; scan++ ) {
            begins[scan] = scan;
            ends[scan] = scan + width * ( height - 1 );
          }
        } else if ( 3 == dir ) {
          // RIGHT:
          for ( int scan=0; scan<agentNum; scan++ ) {
            begins[scan] = scan * width;
            ends[scan] = scan * width + width - 1;
          }
        }
        
        HANDLE_ERROR( cudaMemcpy( devBegins, begins, sizeof(int) * agentNum, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpy( devEnds, ends, sizeof(int) * agentNum, cudaMemcpyHostToDevice ) );
        


        // call Kernel Function 1
        int blockSize = 8;
        dim3 blockDim( blockSize, K );
        int shMemSizePerBlock= blockSize * K * sizeof(float);
        UpdateMessage_device<<<(agentNum/blockSize)+1,blockDim, shMemSizePerBlock>>>( agentNum, 
                                                                                      devD,
                                                                                      devMsg,
                                                                                      devDistance,
                                                                                      devBegins, 
                                                                                      devEnds,
                                                                                      K, 
                                                                                      options.lambda, 
                                                                                      dim, dir, inc[dir],
                                                                                      incK[dir],
                                                                                      area );
        
        HANDLE_ERROR( cudaDeviceSynchronize() );


        
        NormalizeMsg_device<<<(width*height*4)/128+1,128>>>( width * height * 4,
                                                             devMsg,
                                                             K );
        if ( 1 <= options.verbose ) {
          UpdateResult_device<<<(width*height+1)/64,64>>>( width * height,
                                                           devD,
                                                           devMsg,
                                                           devResult,
                                                           K );

          
          HANDLE_ERROR( cudaMemcpy( result, devResult, sizeof(int) * width * height, cudaMemcpyDeviceToHost ) );
          energy = UpdateEnergy( D, label, height, width, K, dim, options.lambda, result );
          printf( "Iteration %d: energy = %.5lf\n", iter, energy );
        }
        
        
        
      } // end for dir
      
    }

    UpdateResult_device<<<(width*height+1)/64,64>>>( width * height,
                                                     devD,
                                                     devMsg,
                                                     devResult,
                                                     K );

    HANDLE_ERROR( cudaMemcpy( result, devResult, sizeof(int) * width * height, cudaMemcpyDeviceToHost ) );
    energy = UpdateEnergy( D, label, height, width, K, dim, options.lambda, result );
    printf( "Final energy = %.5lf\n", energy );

    // Free Cuda Memory
    if ( devD ) HANDLE_ERROR( cudaFree( devD ) );
    if ( devLabel ) HANDLE_ERROR( cudaFree( devLabel ) );
    if ( devMsg ) HANDLE_ERROR( cudaFree( devMsg ) );
    if ( devBegins ) HANDLE_ERROR( cudaFree( devBegins ) );
    if ( devEnds ) HANDLE_ERROR( cudaFree( devEnds ) );
    if ( devResult ) HANDLE_ERROR( cudaFree( devResult ) );
    if ( devDistance ) HANDLE_ERROR( cudaFree( devDistance ) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    
    return energy;
  }
};



