/*********************************************************************************
 * File: BP.hpp
 * Description: Distance Transform based Loopy Belief Propagation for Ising Model
 * by BreakDS, @ University of Wisconsin-Madison, Thu Aug 16 23:23:48 CDT 2012
 *********************************************************************************/

#pragma once

#include <type_traits>
#include <cassert>
#include <cstring>
#include <vector>
#include "LLPack/utils/extio.hpp"
#include "LLPack/utils/SafeOP.hpp"
#include "DT.hpp"
#include "LLPack/utils/debug.hpp"

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

  

  namespace
  {
    void SaveOutResult( const int *result, int n, const std::string& filename )
    {
      WITH_OPEN( out, filename.c_str(), "w" );
      fwrite( result, sizeof(int), n, out );
      END_WITH( out );
    }
    void LoadInResult( int *result, int n, const std::string& filename )
    {
      WITH_OPEN( in, filename.c_str(), "r" );
      fread( result, sizeof(int), n, in );
      END_WITH( in );
    }

    template <typename floating=float>
    inline void NormalizeMessages( floating ** msg, int height, int width, int K )
    {
      const int area = height * width;
      for ( int dir=0; dir <4; dir ++ ) {
        floating *msgp = msg[dir];
        for ( int i=0; i<area; i++ ) {
          floating sum = 0.0;
          for ( int k=0; k<K; k++ ) {
            sum += msgp[k];
          }
          sum /= K;
          for ( int k=0; k<K; k++ ) {
            *(msgp++) -= sum;
          }
        }
      }
    }

    template <typename floating=float>
    double UpdateResult( const floating* D, const floating* label, floating** msg, int* result, 
                         int K, int dim, int height, int width, floating lambda )
    {

      // constants
      const int inc[4] = { -width, -1, width, 1 };
      const int incDim[4] = {-width*K*dim,-K*dim,width*K*dim,K*dim};
      

      // Update result
      const int area = height * width;
      const floating *Dp = D;
      floating *msgp[4];
      for ( int dir=0; dir<4; dir++ ) msgp[dir] = msg[dir];
      for ( int i=0; i<area; i++ ) {
        result[i] = 0;
        floating min = 0;
        for ( int k=0; k<K; k++ ) {
          floating sum = Dp[k];
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

      return energy;
    }
  }
  
  
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
    const int incK[4] = {-width*K,-K,width*K,K};
    const int incDim[4] = {-width*K*dim,-K*dim,width*K*dim,K*dim};
    const int begin[4] = {height-1,width-1,0,0};
    const int end[4] = {0,0,height-1,width-1};
    const int step[4] = {-1,-1,1,1};
    const int order[4] = {2,0,3,1}; // DOWN, UP, RIGHT, LEFT

    // debugging:
    //    debugger::Recorder<true> rec( "debug.dat" );

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
    

    // Initialize Result
    memset( result, 0, sizeof( height * width ) );
    
    for ( int iter=0; iter<options.maxIter; iter++ ) {
      // Update 4 directions sequentially
      for ( int dirID=0; dirID<4; dirID++ ) {
        int dir = order[dirID];
        int opp = (dir+2) & 3;
        int scanBegin = 0;
        int scanEnd = ( 0 == ( dir & 1 ) ) ? width : height;
        for ( int scan=scanBegin; scan<scanEnd; scan++ ) { // outer loop
          const floating *Dp = nullptr;
          const floating *labelp = nullptr;
          int msgp = 0;
          if ( 0 == ( dir & 1 ) ) {
            Dp = D + ( begin[dir] * width + scan ) * K;
            labelp = label + (begin[dir] * width + scan ) * K * dim;
            msgp = ( begin[dir] * width + scan ) * K;
          } else {
            Dp = D + ( scan * width + begin[dir] ) * K;
            labelp = label + ( scan * width + begin[dir] ) * K * dim;
            msgp = ( scan * width + begin[dir] ) * K;
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
            
            // Non Distance-Transformation version
            for ( int k=0; k<K; k++ ) {
              floating min = 0.0;
              for ( int k0=0; k0<K; k0++ ) {
                floating value = 0.0;
                const floating *lp0 = labelp + k0 * dim;
                const floating *lp1 = labelp + incDim[dir] + dim * k;
                for ( int d=0; d<dim; d++ ) {
                  if( lp0[d] > lp1[d] ) {
                    value += (lp0[d] - lp1[d]);
                  } else {
                    value += (lp1[d] - lp0[d]);
                  }
                }
                value *= lambda;
                value += h[k0];
                if ( 0 == k0 || value < min ) min = value;
              }
              msg[dir][msgout+k] = min;
            }
            
            Dp += incK[dir];
            labelp += incDim[dir];
            msgp += incK[dir];
          } // end inner loop (i)
        } // end outer loop (scan)


        
        double energy = UpdateResult<floating>( D, label, msg, result, K, dim, height, width, lambda );
        

        if ( 1 <= options.verbose ) {
          // Energy Function Value
          printf( "Iteration %d: energy = %.5lf\n", iter, energy );
        }

        NormalizeMessages( msg, height, width, K );
        

        
      } // end for dir
    }

    // free the internally created buffer    
    if ( nullptr == msgBuf ) {
      DeleteToNullWithTestArray( buf );
    }
  }
};
