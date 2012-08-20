/*********************************************************************************
 * File: BP.hpp
 * Description: Distance Transform based Belief Propagation for Ising Model
 * by BreakDS, @ University of Wisconsin-Madison, Thu Aug 16 23:23:48 CDT 2012
 *********************************************************************************/

#pragma once

#include <type_traits>
#include <cassert>
#include <cstring>
#include <vector>
#include "LLPack/utils/SafeOP.hpp"
#include "DT.hpp"

namespace optimize
{
  
  

  // Abstract Random Hash Functor
  template <typename floating=float>
  class AbstractRandHash
  {
  public:
    virtual void shuffle( int dim ) = 0;
    virtual floating operator()( const floating* a, int dim) = 0;
  };


  // Optimization Options
  struct Options
  {
    int maxIter; // LoopyBP
    int numHypo; // LoopyBP
    int verbose; // LoopyBP (1=showEngergy and iteration)
    
  };

  
  
  
  /*
   * Loopy Belief Propogation for Ising model,
   * min_{f_i} sum_i  D(f_i)  + lambda * sum_{i,j} ||g_{f_i} - g_{f_j}||
   * where
   * D is a ( height * width * K ) 3-D matrix = Data term
   * labels is a ( height * width * K * dim ) 4-D matrix = label feature
   * msg[dir] is a ( height * width * K ) 3-D matrix = message from dir
   * result is a height * width matrix = the selected indices for each pixel
   */
  template <typename RandHash, typename DistTrans, typename floating=float>
  void LoopyBP( const floating* D, const floating *label, const floating lambda,
                int height, int width, int K, int dim,
                int* result, Options options, floating* msgBuf = nullptr )
  {
    static_assert( std::is_base_of<AbstractDT<floating>,DistTrans>::value,
                   "DistTrans is not derived from AbstractDT." );
    static_assert( std::is_base_of<AbstractRandHash<floating>,RandHash>::value,
                   "RandHash is not derived from AbastractRandHash." );
                   
    // constants:
    // static const int UP = 0;
    // static const int LEFT = 1;
    // static const int DOWN = 2;
    // static const int RIGHT = 3;
    const int inc[4] = { -width, -1, width, 1 };
    const int incK[4] = {-width*K,-K,width*K,K};
    const int incDim[4] = {-width*K*dim,-K*dim,width*K*dim,K*dim};
    const int begin[4] = {height-1,width-1,0,0};
    const int end[4] = {1,1,height-2,width-2};
    const int step[4] = {-1,-1,1,1};


    // Functors
    RandHash hash;
    DistTrans dtrans;

    

    floating *buf = msgBuf; // msg stores messages for each pixel, 4 * height * width * K
    if ( nullptr == buf ) {
      // Message buffer is not provided externally
      buf = new floating[4*width*height*K];
    }
    // Init all message to 0
    memset( buf, 0, sizeof(floating) * 4 * height * width * K );

    int area = height * width;
    floating *msg[4]; // 0 = up, 1 = left, 2 = right, 3 = down
    msg[0] = buf;
    for ( int i=1; i<=3; i++ ) {
      msg[i] = msg[i-1] + area * K;
    }


    // Distance Tranform related Containers:
    vector<floating> h;
    h.resize(K);
    vector<floating> a;
    a.resize(K);
    vector<floating> b;
    b.resize(K);
    vector<int> match;
    match.resize(K);
    
    
    for ( int iter=0; iter<options.maxIter; iter++ ) {
      // Update 4 directions sequentially
      for ( int dir=0; dir<4; dir++ ) {
        int opp = (dir+2) & 3;
        int scanBegin = begin[3-dir];
        int scanEnd = end[3-dir];
        int movement = step[3-dir];
        for ( int scan=scanBegin; scan!=scanEnd; scan+=movement ) { // outer loop
          const floating *Dp = nullptr;
          const floating *labelp = nullptr;
          int msgp = 0;
          if ( 0 == ( dir & 1 ) ) {
            Dp = D + ( begin[dir] * width + scanBegin ) * K;
            labelp = label + (begin[dir] * width + scanBegin ) * K * dim;
            msgp = ( begin[dir] * width + scanBegin ) * K;
          } else {
            Dp = D + ( scanBegin * width + begin[dir] ) * K;
            labelp = label + ( scanBegin * width + begin[dir] ) * K * dim;
            msgp = ( scanBegin * width + begin[dir] ) * K;
          }
          
          int forward = step[dir];
          for ( int i=begin[dir]; i!=end[dir]; i+=forward ) { // inner loop
            /*
             * Message Passing Formulation:
             * a = pixel index
             * b = pixel index (neighbor)
             * f_a = feature of a-th pixel
             * q_{a,i} = feature of i-th candidates for a-th pixel
             * l_{a,i } = labels of i-th candidates for a-th pixel
             * N(a) = neighbor of a
             * m_{a->b}(j) = min_{i} lambda * ||l_{b,j}-l_{a,i}|| + ||q_{a,i}-f_a||^2 + message(N(a)/b)
             *             = lambda * min_{i} ||b[j] - a[i]|| + 1/lambda * (D(i) + message(N(a)/b))
             *             = lambda * min_{i} ||b[j] - a[i]|| + h[i] / lambda
             */
            int msgout = msgp + incK[dir];

            // Preparation for Distance Transform
            for ( int k=0; k<K; k++ ) {
              h[k] = Dp[k];
              for ( int j=0; j<4; j++ ) {
                // TODO: use more clever summing strategy
                if ( j != opp ) {
                  h[k] += msg[j][msgp+k];
                }
              }
              h[k] /= lambda;
            }

            /* // Fast Distance Transformation Version
            for ( int hypo=0; hypo<options.numHypo; hypo++ ) {
              // Hypothesis Generation
              hash.shuffle(dim);

              const floating *lp = labelp;
              for ( int k=0; k<K; k++ ) {
                a[k] = hash( lp, dim );
                lp += dim;
              }
              lp = labelp + incDim[dir];
              for ( int k=0; k<K; k++ ) {
                b[k] = hash( lp, dim );
                lp += dim;
              }

              // Distance Transform
              dtrans( h, a, b, K, match );

              // Update Message
              for ( int k=0; k<K; k++ ) {
                const floating *lp0 = labelp + match[k] * dim;
                const floating *lp1 = labelp + incDim[dir] + dim * k;
                floating message = h[match[k]];
                for ( int d=0; d<dim; d++ ) {
                  if ( lp0[d] > lp1[d] ) {
                    message += (lp0[d] - lp1[d]);
                  } else {
                    message += (lp1[d] - lp0[d]);
                  }
                }
                if ( 0 == hypo || message < msg[dir][msgout+k] ) msg[dir][msgout+k] = message;
              } // end for k
            } // end for hypo
            */
            
            // Non Distance Transformation version
            for ( int k=0; k<K; k++ ) {
              floating min = 0.0;
              for ( int k0=0; k0<K; k0++ ) {
                floating value = h[k0];
                const floating *lp0 = labelp + k0 * dim;
                const floating *lp1 = labelp + incDim[dir] + dim * k;
                for ( int d=0; d<dim; d++ ) {
                  if( lp0[d] > lp1[d] ) {
                    value += (lp0[d] - lp1[d]);
                  } else {
                    value += (lp1[d] - lp0[d]);
                  }
                }
                if ( 0 == k0 || value < min ) min = value;
              }
              msg[dir][msgout+k] = min * lambda;
            }
            
            Dp += incK[dir];
            labelp += incDim[dir];
            msgp += incK[dir];
          } // end inner loop
        } // end outer loop
      } // end for dir


      printf( "D(0,1)=%.4f\n", D[1] );
      

      // Energy Function Value
      if ( 1 <= options.verbose ) {
        // Update result
        const floating *Dp = D;
        floating *msgp[4];
        for ( int dir=0; dir<4; dir++ ) msgp[dir] = msg[dir];
        for ( int i=0; i<area; i++ ) {
          result[i] = 0;
          floating min = 0;
          for ( int k=0; k<K; k++ ) {
            floating sum = Dp[k];
            // if ( 333 == i ) {
            printf( "k=%d ", k );
            printf( "D=%.4f ", sum );
            // }
            for ( int dir=0; dir<4; dir++ ) {
              sum += msgp[dir][k];
            }
            // if ( 333 == i ) {
            printf( "sum=%.4f\n", sum );
            char ch;
            scanf( "%c", &ch );
            // }
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
            energy += D[i];

            // UP:
            int d = 0;
            if ( y > 0 ) {
              const float *lp0 = labelp + result[i] * dim;
              const float *lp1 = labelp + incDim[d] + result[i+inc[d]] * dim;
              float l1 = 0.0;
              for ( int di=0; di<dim; di++ ) {
                if ( *(lp0) > *(lp1) ) {
                  l1 += ( *lp0 - *lp1 );
                } else { 
                  l1 += ( *lp1 - *lp0 );
                }
                lp0++;
                lp1++;
              }
              energy += l1 * lambda;
            }


            // LEFT:
            d = 1;
            if ( x > 0 ) {
              const float *lp0 = labelp + result[i] * dim;
              const float *lp1 = labelp + incDim[d] + result[i+inc[d]] * dim;
              float l1 = 0.0;
              for ( int di=0; di<dim; di++ ) {
                if ( *(lp0) > *(lp1) ) {
                  l1 += ( *lp0 - *lp1 );
                } else { 
                  l1 += ( *lp1 - *lp0 );
                }
                lp0++;
                lp1++;
              }
              energy += l1 * lambda;
            }
            
            i++;
            labelp += K * dim;
          }
        }
        
        printf( "Iteration %d: energy = %.5lf\n", iter, energy );
      } // end of if verbose >= 1
    }
    
    
    // free the internally created buffer    
    if ( nullptr == msgBuf ) {
      DeleteToNullWithTestArray( buf );
    }
  }
};
