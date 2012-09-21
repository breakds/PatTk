/*********************************************************************************
 * File: BP_CUDA.cu
 * Description: Belief Propagation for Ising Model, CUDA parallelized version
 * by BreakDS, @ University of Wisconsin-Madison, Fri Aug 31 15:25:19 CDT 2012
 *********************************************************************************/

#include "BP_CUDA.h"
#include <cuda.h>
#include "../utils/cuda_aux.hpp"



// Temporary Constants
#define PATCH_SIDE 17

__device__ __constant__ int g_inc[4];
  

namespace optimize_cuda
{

  /*
  __host__ __device__ inline float dist( const float *a, const float *b,
                                int __attribute__((__unused__)) dim,
                                int direction = -1 )
  {
    const float coeff[6] = { 0.0, 30.0, 30.0, 10.0, 1.0, 1.0 };
    float tmp;
    tmp = ( a[0] > b[0] ) ? ( a[0] - b[0] ) : ( b[0] - a[0] );


    // [0] = image index
    tmp = ( a[0] > b[0] ) ? ( a[0] - b[0] ) : ( b[0] - a[0] );
    if ( tmp > 1.0 ) return 150000.0;

    // [1], [2] = dy, dx (rotation representation)
    tmp = ( a[1] > b[1] ) ? ( a[1] - b[1] ) : ( b[1] - a[1] );
    tmp += ( a[2] > b[2] ) ? ( a[2] - b[2] ) : ( b[2] - a[2] );
    if ( tmp > 1.0 ) {
      return 150000.0;
    }

      
    // [4],[5] = spatial distance
    // Should be compensated by the rotation (dy,dx)
    // Note that dy = b[1] dx = b[2] by definition
    float ay(a[4]), ax(a[5]);
    if ( 0 == direction ) {
      ay += b[2];
      ax -= b[1];
    } else if ( 1 == direction ) {
      ay += b[1];
      ax -= b[2];
    } else if ( 2 == direction ) {
      ay += b[2];
      ax += b[1];
    } else if ( 3 == direction ) {
      ay -= b[1];
      ax += b[2];
    }
      
    tmp = ( ay > b[4] ) ? ( ay - b[4] ) : ( b[4] - ay );
    tmp += ( ax > b[5] ) ? ( ax - b[5] ) : ( b[5] - ax );
    if ( tmp > PATCH_SIDE ) {
      return 150000.0;
    }
      
    // [1],[2] = spatial distance
    
    float sum = tmp * coeff[4];
    
    for ( int i=1; i<4; i++ ) {
      sum += ( ( a[i] > b[i] ) ? ( a[i] - b[i] ) : ( b[i] - a[i] ) ) * coeff[i];
    }
    return sum;
  }
  */
  

  __host__ __device__ inline float dist( const float a0, const float a1,
                                         const float a2, const float a3,
                                         const float a4, const float a5,
                                         const float b0, const float b1,
                                         const float b2, const float b3,
                                         const float b4, const float b5,
                                         int __attribute__((__unused__)) dim,
                                         int direction = -1 )
  {
    const float coeff[6] = { 0.0, 30.0, 30.0, 10.0, 1.0, 1.0 };
    float tmp;

    // [0] = image index
    tmp = fabsf( b0 - a0 );
    if ( tmp > 1.0 ) return 150000.0;

    // [1], [2] = dy, dx (rotation representation)
    tmp = fabsf( a1 - b1 ) + fabsf( a2 - b2  );
    if ( tmp > 1.0 ) {
      return 150000.0;
    }

      
    // [4],[5] = spatial distance
    // Should be compensated by the rotation (dy,dx)
    // Note that dy = b1 dx = b2 by definition

    float ay(a4), ax(a5);

    if ( 0 == direction ) {
      ay += b2;
      ax -= b1;
    } else if ( 1 == direction ) {
      ay += b1;
      ax -= b2;
    } else if ( 2 == direction ) {
      ay += b2;
      ax += b1;
    } else if ( 3 == direction ) {
      ay -= b1;
      ax += b2;
    }
    
    
    
      
    tmp = fabsf( ay - b4 ) + fabsf( ax - b5 );
    if ( tmp > PATCH_SIDE ) {
      return 150000.0;
    }
      
    // [1],[2] = spatial distance
    
    float sum = tmp * coeff[4];

    sum += fabsf( b1 - a1 ) * coeff[1];
    sum += fabsf( b2 - a2 ) * coeff[2];
    sum += fabsf( b3 - a3 ) * coeff[3];

    
    return sum;
  }
  


  __global__ void NormalizeMessages_float_agent( const int agentNum,
                                                 float *msg,
                                                 int K )
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      float avg = 0.0f;
      for ( int k=0; k<K; k++ ) {
        avg += msg[ idx * K + k ];
        // avg += msg[ k * agentNum + idx ];
      }
      avg /= K;
      for ( int k=0; k<K; k++ ) {
        msg[ idx * K + k ] -= avg;
        // msg[ k * agentNum + idx ] -= avg;
      }
    }
  }

  
  
  __global__ void UpdateResult_float_agent( const int agentNum, const float *D,
                                            const float* msg,
                                            int *result, int K, int dim, float lambda )
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if ( idx < agentNum ) {
      result[idx] = 0;
      float min = 0.0f;
      for ( int k=0; k<K; k++ ) {
        float sum = D[idx*K+k];
        for ( int dir=0; dir<4; dir++ ) {
          sum += msg[ ( dir * agentNum + idx ) * K + k ];
          // sum += msg[ ( k * 4  + dir ) * agentNum + idx ];
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
          // energy += dist( lp0, lp1, dim, d ) * lambda;
          energy += dist( lp0[0], lp0[1], lp0[2], lp0[3], lp0[4], lp0[5],
                          lp1[0], lp1[1], lp1[2], lp1[3], lp1[4], lp1[5],
                          dim, d ) * lambda;
        }
        

        // LEFT:
        d = 1;
        if ( x > 0 ) {
          const float *lp0 = labelp + result[i] * dim;
          const float *lp1 = labelp + incDim[d] + result[i+inc[d]] * dim;
          // energy += dist( lp0, lp1, dim, d ) * lambda;
          energy += dist( lp0[0], lp0[1], lp0[2], lp0[3], lp0[4], lp0[5],
                          lp1[0], lp1[1], lp1[2], lp1[3], lp1[4], lp1[5],
                          dim, d ) * lambda;
        }

        i++;
        labelp += K * dim;
      }
    }

    return energy;
  }

  /*
  __global__ void LoopyBP_round1( int agentNum, // number of agents needes
                                  int iter,
                                  float *D,
                                  float* msg,
                                  float *distance,
                                  int* begins,
                                  int inc,
                                  float *h_buf,
                                  int K,
                                  int dir,
                                  int area )
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      int k = threadIdx.y;
      int opp = (dir+2) & 3;
      int pos = k * agentNum + idx;
      int pixel = begins[idx] + inc * iter;
      float tmp = D[pixel * K + k];
      for ( int j=0; j<4; j++ ) {
        if ( j != opp ) {
          // h_buf[pos] += msg[ j * area * K + pixel * K + k];
          tmp += msg[ ( k * 4 + j ) * area + pixel ];
        }
      }
      h_buf[pos] = tmp;
    }
  }
  
  __global__ void LoopyBP_round2( int agentNum, // number of agents needes
                                  int iter,
                                  float *D,
                                  float* msg,
                                  float *distance,
                                  int* begins,
                                  int inc,
                                  float *h_buf,
                                  int K,
                                  float lambda,
                                  int dir,
                                  int area )
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      int k = threadIdx.y;
      int pos = k * agentNum + idx;
      int pixel = begins[idx] + inc * iter;
      float min = distance[ ( k * 4  + dir ) * area + pixel ] * lambda + h_buf[pos];
      for ( int k0=1; k0<K; k0++ ) {
        float value = distance[ ( ( k0 * K + k ) * 4 + dir ) * area + pixel ] * lambda +
          h_buf[pos + k0 * agentNum];
        if ( value < min ) min = value;
      }
      // msg[ dir * area * K + ( pixel + inc ) * K + k ] = min;
      msg[ ( k * 4 + dir ) * area + pixel + inc ] = min;
    }
  }
  
  */

  
  // Kernel Function for loopy belief propagation (agentNum = width/height)
  __global__ void LoopyBP_agent_float( int agentNum, // number of agents needes
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
                                       int incDim,
                                       int area )
  {
    
    extern __shared__ float buf[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      float *h = buf;
      // float *l0 = buf + blockDim.x * K + threadIdx.x * dim;
      // float *l1 = buf + blockDim.x * ( K + dim ) + threadIdx.x * dim;
      // agent initialization
      int opp = (dir+2) & 3;
      //      float *h = buf_h + idx * K;
      float *Dp = D + begins[idx] * K;
      
      for ( int i=begins[idx]; i!=ends[idx]; i+=inc, Dp+=incK ) {
        
        for ( int k=0; k<K; k++ ) {
          h[ k * blockDim.x + threadIdx.x ] = Dp[k];
          for ( int j=0; j<4; j++ ) {
            if ( j != opp ) {
              h[ k * blockDim.x + threadIdx.x ] += msg[ j * area * K + i * K + k ];
              // h[ k * blockDim.x + threadIdx.x ] += msg[ ( k * 4 + j ) * area + i ];
            }
          }
        }

        for ( int k=0; k<K; k++ ) {
          float min = distance[ ( k * 4 + dir ) * area  + i ] * lambda + h[ threadIdx.x ];
          
          for ( int k0=1; k0<K; k0++ ) {
            float value = distance[ ( ( k0 * K + k ) * 4 + dir ) * area  + i ] * lambda +
              h[ k0 * blockDim.x + threadIdx.x];
            if ( value < min ) min = value;
          }
          msg[ dir * area * K + ( i + inc ) * K + k ] = min;
          // msg[ ( k * 4 + dir ) * area + i + inc ] = min;
        }

      } // end for i
    }
  }

  __global__ void LoopyBP_agent_float_v2( int agentNum, // number of agents needes
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
                                          int incDim,
                                          int area )
  {
    
    extern __shared__ float buf[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      int k = threadIdx.y;
      float *h = buf;
      // float *l0 = buf + blockDim.x * K + threadIdx.x * dim;
      // float *l1 = buf + blockDim.x * ( K + dim ) + threadIdx.x * dim;
      // agent initialization
      int opp = (dir+2) & 3;
      //      float *h = buf_h + idx * K;
      float *Dp = D + begins[idx] * K;
      
      for ( int i=begins[idx]; i!=ends[idx]; i+=inc, Dp+=incK ) {
        
        h[ k * blockDim.x + threadIdx.x ] = Dp[k];
        for ( int j=0; j<4; j++ ) {
          if ( j != opp ) {
            h[ k * blockDim.x + threadIdx.x ] += msg[ j * area * K + i * K + k ];
            // h[ k * blockDim.x + threadIdx.x ] += msg[ ( k * 4 + j ) * area + i ];
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
        // msg[ ( k * 4 + dir ) * area + i + inc ] = min;

      } // end for i
    }
  }


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
      
      distance[idx] = dist( lp0[0], lp0[1], lp0[2], lp0[3], lp0[4], lp0[5],
                            lp1[0], lp1[1], lp1[2], lp1[3], lp1[4], lp1[5],
                            dim, dir );
    }
  }




  
  
  double LoopyBP( const float *D, // The distance term for each pixel, h x w
                  const float *label,
                  // the label vectors for each candidates of each pixel, h x w x K x dim
                  const int height,
                  const int width,
                  const int K,
                  const int dim,
                  int *result,
                  Options options, float* msgBuf )
  {
    const int inc[4] = { -width, -1, width, 1 };
    HANDLE_ERROR( cudaMemcpyToSymbol( "g_inc", inc, sizeof(int) * 4, 0, cudaMemcpyHostToDevice ) );
    const int incK[4] = {-width*K,-K,width*K,K};
    const int incDim[4] = {-width*K*dim,-K*dim,width*K*dim,K*dim};
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
    cudaMalloc( (void**) &devD, sizeof(float) * area * K );
    cudaMemcpy( devD, D, sizeof(float) * area * K, cudaMemcpyHostToDevice );

    // Labels array
    float *devLabel = NULL;
    cudaMalloc( (void**) &devLabel, sizeof(float) * area * K * dim );
    cudaMemcpy( devLabel, label, sizeof(float) * area * K * dim, cudaMemcpyHostToDevice );
    
    // Messages
    // After these lines, msg[0] = up, msg[1] = left, msg[2] = right, msg[3] = down
    float *devMsg = NULL;
    cudaMalloc( (void**) &devMsg, sizeof(float) * area * K * 4 );
    cudaMemset( devMsg, 0, sizeof(float) * area * K * 4 );
    

    // begins and ends
    int longer = height > width ? height : width;
    int *devBegins = NULL;
    int *devEnds = NULL;
    int begins[longer];
    int ends[longer];
    cudaMalloc( (void**) &devBegins, sizeof(int) * longer );
    cudaMalloc( (void**) &devEnds, sizeof(int) * longer );

    // Result
    int *devResult = NULL;
    cudaMalloc( (void**) &devResult, sizeof(int) * area * K );

    // h buffer
    // float *devH = NULL;
    // cudaMalloc( (void**) &devH, sizeof(float) * longer * K );


    // Intermediate Distance Calculation
    float *devDistance = NULL;
    HANDLE_ERROR( cudaMalloc( (void**) &devDistance, sizeof(float) * K * K * 4 * area ) );

    dim3 precompute_grid( ( area + 1 ) / 8, 4 );
    dim3 precompute_block( 8, K, K );
    


    // uint kernelTime;
    // cutCreateTimer(&kernelTime);
    // cutResetTimer(kernelTime);
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
        int length = 0;
        if ( 0 == dir ) {
          // UP:
          for ( int scan=0; scan<agentNum; scan++ ) {
            begins[scan] = scan + width * ( height - 1 );
            ends[scan] = scan;
            length = height - 1;
          }
        } else if ( 1 == dir ) {
          // LEFT:
          for ( int scan=0; scan<agentNum; scan++ ) {
            begins[scan] = width * scan + width - 1;
            ends[scan] = width * scan;
            length = width - 1;
          }
        } else if ( 2 == dir ) {
          // DOWN:
          for ( int scan=0; scan<agentNum; scan++ ) {
            begins[scan] = scan;
            ends[scan] = scan + width * ( height - 1 );
            length = height - 1;
          }
        } else if ( 3 == dir ) {
          // RIGHT:
          for ( int scan=0; scan<agentNum; scan++ ) {
            begins[scan] = scan * width;
            ends[scan] = scan * width + width - 1;
            length = width - 1;
          }
        }
        
        cudaMemcpy( devBegins, begins, sizeof(int) * agentNum, cudaMemcpyHostToDevice );
        cudaMemcpy( devEnds, ends, sizeof(int) * agentNum, cudaMemcpyHostToDevice );
        


        // Alternative Calling
        // int blockSize = 16;
        // dim3 blockDim( blockSize, K );
        // for ( int i=0; i<length; i++ ) {
        //   LoopyBP_round1<<<(agentNum/blockSize)+1, blockDim>>>( agentNum,
        //                                                         i,
        //                                                         devD,
        //                                                         devMsg,
        //                                                         devDistance,
        //                                                         devBegins,
        //                                                         inc[dir],
        //                                                         devH,
        //                                                         K,
        //                                                         dir,
        //                                                         area );
        //   LoopyBP_round2<<<(agentNum/blockSize)+1, blockDim>>>( agentNum,
        //                                                         i,
        //                                                         devD,
        //                                                         devMsg,
        //                                                         devDistance,
        //                                                         devBegins,
        //                                                         inc[dir],
        //                                                         devH,
        //                                                         K,
        //                                                         options.lambda,
        //                                                         dir,
        //                                                         area );
        // }

        // call Kernel Function 1
        int blockSize = 8;
        dim3 blockDim( blockSize, K );
        int shMemSizePerBlock= blockSize * K * sizeof(float);
        LoopyBP_agent_float_v2<<<(agentNum/blockSize)+1,blockDim, shMemSizePerBlock>>>( agentNum, 
                                                                                        devD,
                                                                                        devMsg,
                                                                                        devDistance,
                                                                                        devBegins, 
                                                                                        devEnds,
                                                                                        K, 
                                                                                        options.lambda, 
                                                                                        dim, dir, inc[dir],
                                                                                        incK[dir], incDim[dir],
                                                                                        area );
        
        HANDLE_ERROR( cudaDeviceSynchronize() );


        
        // cudaMemcpy( result, devResult, sizeof(int) * width * height, cudaMemcpyDeviceToHost );

        // energy = UpdateEnergy( D, label, height, width, K, dim, options.lambda, result );
        
        NormalizeMessages_float_agent<<<(width*height*4)/128+1,128>>>( width * height * 4,
                                                                       devMsg,
                                                                       K );
        if ( 1 <= options.verbose ) {

          UpdateResult_float_agent<<<(width*height+1)/64,64>>>( width * height,
                                                                devD,
                                                                devMsg,
                                                                devResult,
                                                                K,
                                                                dim,
                                                                options.lambda );
          
          cudaMemcpy( result, devResult, sizeof(int) * width * height, cudaMemcpyDeviceToHost );
          energy = UpdateEnergy( D, label, height, width, K, dim, options.lambda, result );
          printf( "Iteration %d: energy = %.5lf\n", iter, energy );
        }
        
        

      } // end for dir
      
    }

    UpdateResult_float_agent<<<(width*height+1)/64,64>>>( width * height,
                                                          devD,
                                                          devMsg,
                                                          devResult,
                                                          K,
                                                          dim,
                                                          options.lambda );

    cudaMemcpy( result, devResult, sizeof(int) * width * height, cudaMemcpyDeviceToHost );
    energy = UpdateEnergy( D, label, height, width, K, dim, options.lambda, result );
    printf( "Final energy = %.5lf\n", energy );

    // Free Cuda Memory
    if ( devD ) cudaFree( devD );
    if ( devLabel ) cudaFree( devLabel );
    if ( devMsg ) cudaFree( devMsg );
    if ( devBegins ) cudaFree( devBegins );
    if ( devEnds ) cudaFree( devEnds );
    if ( devResult ) cudaFree( devResult );
    if ( devDistance ) cudaFree( devDistance );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    
    return energy;
  }
};



