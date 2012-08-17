/*********************************************************************************
 * File: BP.hpp
 * Description: Distance Transform based Belief Propagation for Ising Model
 * by BreakDS, @ University of Wisconsin-Madison, Thu Aug 16 23:23:48 CDT 2012
 *********************************************************************************/

#pragma once

#include <type_traits>
#include <cassert>
#include <vector>
#include "LLPack/utils/SafeOP.hpp"
#include "DT.hpp"

namespace optimize
{

  struct Options
  {
    int maxIter; // LoopyBP
    int numHypo; // LoopyBP
  };
  /*
   * Loopy Belief Propogation for Ising model,
   * min_{f_i} sum_i  D(f_i)  + lambda * sum_{i,j} ||g_{f_i} - g_{f_j}||
   * where
   * D is a ( width * height * K ) 3-D matrix = Data term
   * labels is a ( width * height * K * dim ) 4-D matrix = label feature
   * msg[dir] is a ( width * height * K ) 3-D matrix = message from dir
   * result is a width * height matrix = the selected indices for each pixel
   */
  template <typename RandHash, typename DistTrans, typename floating=float>
  void LoopyBP( const floating* D, const floating *label, const floating lambda,
                int height, int width, int K, int dim,
                int* result, Options options, floating* msgBuf = nullptr )
  {
    // constants:
    static const int UP = 0;
    static const int LEFT = 1;
    static const int DOWN = 2;
    static const int RIGHT = 3;
    const int inc[4] = {-width,-1,width,1};
    const int incK[4] = {-width*K,-K,width*K,K};
    const int incDim[4] = {-width*K*dim,-K*dim,width*K*dim,K*dim};
    const int begin[4] = {height-1,width-1,0,0};
    const int end[4] = {1,1,height-2,width-2};


    // Functors
    RandHash hash;
    DistTrans dtrans;

    

    floating *buf = msgBuf; // msg stores messages for each pixel, 4 * width * height * K
    if ( nullptr == buf ) {
      // Message buffer is not provided externally
      buf = new floating[4*width*height*K];
    }
    // Init all message to 0
    memset( buf, 0, sizeof(floating) * 4 * width * height * K );

    int area = width * height;
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
      for ( int dir=0; dir<4; dir++ ) {
        int opp = (dir+2) & 3;
        int scanBegin = begin[3-dir];
        int scanEnd = end[3-dir];
        for ( int scan=scanBegin; scan<scanEnd; scan++ ) { // outer loop
          floating *Dp = nullptr;
          floating *labelp = nullptr;
          int msgp = 0;
          if ( 0 == dir & 1 ) {
            Dp = D + ( begin[dir] * width + scanBegin ) * K;
            labelp = label + (begin[dir] * width + scanBegin ) * K * dim;
            msgp = ( begin[dir] * width + scanBegin ) * K;
          } else {
            Dp = D + ( scanBegin + begin[dir] * width ) * K;
            labelp = label + ( scanBegin + begin[dir] * width ) * K * dim;
            msgp = ( scanBegin + begin[dir] * width ) * K;
          }
          for ( int i=begin[dir]; i<end[dir]; i++ ) {
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
            
            for ( int hypo=0; hypo<options.numHypo; hypo++ ) {
              // Hypothesis Generation
              hash.shuffle();
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
              floating *lp = labelp;
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
              int msgout = msgp + incK[dir];
              for ( int k=0; k<K; k++ ) {
                floating *lp0 = labelp + match[k] * dim;
                floating *lp1 = labelp + incDim[dim] + dim * k;
                floating message = h[match[k]];
                for ( int d=0; d<dim; d++ ) {
                  if ( lp0[d] > lp1[d] ) {
                    message += (lp0[d] - lp1[d]);
                  } else {
                    message += (lp1[d] - lp0[d]);
                  }
                }
                if ( 0 == hypo || message < msg[dir][msgout+k] ) msg[dir][msgout] = message;
              }
            }
          }
        }
      }
    }

    // Fill result
    floating *Dp = D;
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

    
    // free the internally created buffer    
    if ( nullptr == msgBuf ) {
      DeleteToNullWithTestArray( buf );
    }
  }
};
