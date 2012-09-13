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

namespace optimize_cuda
{

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
    // Not that dy = b[1] dx = b[2]
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



  __global__ void NormalizeMessages_float_agent( const int agentNum,
                                                 float *msg,
                                                 int K )
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      float avg = 0.0f;
      for ( int dir=0; dir<4; dir++ ) {
        for ( int k=0; k<K; k++ ) {
          avg += msg[ ( dir * agentNum + idx ) * K + k ];
        }
        avg /= K;
        for ( int k=0; k<K; k++ ) {
          msg[ ( dir * agentNum + idx ) * K + k ] -= avg;
        }
      }
    }
  }
  
  __global__ void UpdateResult_float_agent( const int agentNum, const float *D, const float *label,
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

  

  /*
  double UpdateResult( const float* D, const float* label, float** msg, int* result, 
                       int K, int dim, int height, int width, float lambda )
                       {
  
    // constants
    const int inc[4] = { -width, -1, width, 1 };
    const int incDim[4] = {-width*K*dim,-K*dim,width*K*dim,K*dim};


    // Update result
    const int area = height * width;
    const float *Dp = D;
    float *msgp[4];
    for ( int dir=0; dir<4; dir++ ) msgp[dir] = msg[dir];

    for ( int i=0; i<area; i++ ) {
      result[i] = 0;
      float min = 0;
      for ( int k=0; k<K; k++ ) {
        float sum = Dp[k];
        for ( int dir=0; dir<4; dir++ ) {
          sum += msgp[dir][k];
        }

        if ( 0 == k ) {
          min = sum;
        } else if ( sum < min ) {
          min = sum;
          result[i] = k;
        }
      }
      Dp += K;
      for ( int dir=0; dir<4; dir++ ) msgp[dir] += K;
    }


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
          energy += dist( lp0, lp1, dim, d ) * lambda;
        }


        // LEFT:
        d = 1;
        if ( x > 0 ) {
          const float *lp0 = labelp + result[i] * dim;
          const float *lp1 = labelp + incDim[d] + result[i+inc[d]] * dim;
          energy += dist( lp0, lp1, dim, d ) * lambda;
        }

        i++;
        labelp += K * dim;
      }
    }

    return energy;
  }
  */

  // Kernel Function for updating the result (agentNum = # pixels)
  /*
    __global__ void UpdateResult_agent_float( int agentNum,
    float *D,
    float *msg[4],
    int K,
    int *result  )
    {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
    float *Dp = D + idx * K;
    float *msgp[4];
    for ( int dir=0; dir<4; dir++ ) msgp[dir] = msg[dir] + idx * K;
    result[idx] = 0;
    floating min = 0;
    for ( int k=0; k<K; k++ ) {
    floating sum = Dp[k];
    for ( int dir=0; dir<4; dir++ ) {
    sum += msgp[dir][k];
    }

    if ( 0 == K ) {
    min = sum;
    } else if ( sum < min ) {
    min = sum;
    result[idx] = k;
    }
    }
    }
    }
  */

  
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
          energy += dist( lp0, lp1, dim, d ) * lambda;
        }
        

        // LEFT:
        d = 1;
        if ( x > 0 ) {
          const float *lp0 = labelp + result[i] * dim;
          const float *lp1 = labelp + incDim[d] + result[i+inc[d]] * dim;
          energy += dist( lp0, lp1, dim, d ) * lambda;
        }

        i++;
        labelp += K * dim;
      }
    }

    return energy;
  }

  



  // Kernel Function for loopy belief propagation (agentNum = width/height)
  __global__ void LoopyBP_agent_float( int agentNum, // number of agents needes
                                       float *D,
                                       float *label,
                                       float* msg,
                                       int *begins,
                                       int *ends,
                                       int K,
                                       float lambda,
                                       int dim,
                                       int dir,
                                       int inc,
                                       int incK,
                                       int incDim,
                                       int area,
                                       float *buf_h )
  {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( idx < agentNum ) {
      // agent initialization
      int opp = (dir+2) & 3;
      //      float *h = buf_h + idx * K;
      float *Dp = D + begins[idx] * K;
      float *labelp = label + begins[idx] * K * dim;

      for ( int i=begins[idx]; i!=ends[idx]; i+=inc, Dp+=incK, labelp+=incDim  ) {
        for ( int k=0; k<K; k++ ) {
          if ( i * K + k > 320 * K ) {
            printf( "%d\n", i*K+k );
          }
          buf_h[i*K+k] = Dp[k];
          for ( int j=0; j<4; j++ ) {
            if ( j != opp ) {
              buf_h[i*K+k] += msg[ j * area * K + i * K + incK ];
            }
          }
        }
        
        for ( int k=0; k<K; k++ ) {
          float min = 0.0;
          for ( int k0=0; k0<K; k0++ ) {
            const float *lp0 = labelp + k0 * dim;
            const float *lp1 = labelp + incDim + dim * k;
            float value = dist( lp0, lp1, dim, dir ) * lambda + buf_h[ i * K + k0 ];
            if ( 0 == k0 || value < min ) min = value;
          }
          msg[ dir * area * K + ( i + inc ) * K + k ] = min;
        }
      } // end for i
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

    // h buffer
    float *devBufH = NULL;
    HANDLE_ERROR( cudaMalloc( (void**) &devBufH, sizeof(float) * longer * K ) );
  
    // Result
    int *devResult = NULL;
    cudaMalloc( (void**) &devResult, sizeof(int) * area * K );




    double energy(0);

    for ( int iter=0; iter<options.maxIter; iter++ ) {
      for ( int dirID=0; dirID<4; dirID++ ) {
        int dir = order[dirID];

        // fill in begins and ends
        int agentNum = ( 0 == ( dir & 1 ) ) ? width : height;
        int range = ( 0 == ( dir & 1 ) ) ? height * width : width;
        int stride = ( 0 == ( dir & 1 ) ) ? 1 : width;
        for ( int scan=0; scan<agentNum; scan++ ) {
          begins[scan] = scan * stride;
          ends[scan] = begins[scan] + range;
        }
        cudaMemcpy( devBegins, begins, sizeof(int) * agentNum, cudaMemcpyHostToDevice );
        cudaMemcpy( devEnds, ends, sizeof(int) * agentNum, cudaMemcpyHostToDevice );

        // Call Kernel Function 1
        LoopyBP_agent_float<<<(agentNum/32)+1,32>>>( agentNum, devD, devLabel,
                                                            devMsg, devBegins, devEnds,
                                                            K, options.lambda, dim, dir, inc[dir],
                                                            incK[dir], incDim[dir],
                                                            area,
                                                            devBufH );

        
        // debugging:
        float *tmpMsg = new float[area * K * 4];
        HANDLE_ERROR( cudaMemcpy( tmpMsg, devMsg, sizeof(float) * area * K * 4, cudaMemcpyDeviceToHost ) );
        float *loadMsg = new float[area * K * 4];
        FILE *in = fopen( "debug.dat", "r" );
        fread( loadMsg, sizeof(float), area * K * 4, in );
        fclose( in );
        for ( int l=0; l<area*K*4; l++ ) {
          if ( fabsf( tmpMsg[l] - loadMsg[l] ) > 1e-5 ) {
            printf( "%d: load(%.5f) vs gen(%.5f)\n", l, loadMsg[l], tmpMsg[l] );
            char ch;
            scanf( "%c", &ch );
          }
        }
        delete[] loadMsg;
        delete[] tmpMsg;
        printf( "debugging done\n" );
        exit( -1 );
        // end debugging


        
        UpdateResult_float_agent<<<(width*height)/32+1,32>>>( width * height,
                                                              devD,
                                                              devLabel,
                                                              devMsg,
                                                              devResult,
                                                              K,
                                                              dim,
                                                              options.lambda );


        cudaMemcpy( result, devResult, sizeof(int) * width * height, cudaMemcpyDeviceToHost );
        
        energy = UpdateEnergy( D, label, height, width, K, dim, options.lambda, result );
        
        NormalizeMessages_float_agent<<<(width*height)/32+1,32>>>( width * height,
                                                                   devMsg,
                                                                   K );

        
        
        /*
            __global__ void LoopyBP_agent_float( int agentNum, // number of agents needes
                                       float *D,
                                       float *label,
                                       float* msg[4],
                                       int *begins,
                                       int *ends,
                                       int K,
                                       float lambda,
                                       int dim,
                                       int dir,
                                       int inc,
                                       int incK,
                                       int incDim,
                                       int area,
                                       float *buf_h )

        */


        if ( 1 <= options.verbose ) {
          printf( "Iteration %d: energy = %.5lf\n", iter, energy );
        }

      } // end for dir
    }

    for ( int i=0; i<10; i++ ) {
      printf( "result[%d] = %d\n", i, result[i] );
    }
    return energy;
  }
};



