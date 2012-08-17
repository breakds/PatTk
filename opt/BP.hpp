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

namespace optimize
{

  struct Options
  {
    maxIter; // LoopyBP
  };
  /*
   * Loopy Belief Propogation for Ising model,
   * min_{f_i} sum_i  D(f_i)  + lambda * sum_{i,j} ||g_{f_i} - g_{f_j}||
   * where
   * D is a ( width * height * K ) 3-D matrix = Data term
   * labels is a ( width * height * K * dim ) 4-D matrix = label feature
   * result is a width * height matrix = the selected indices for each pixel
   */
  template <typename floating=float>
  void LoopyBP( const floating* D, const floating *label, const floating lambda
                int height, int width, int K, int dim,
                int* result, Options options, floating* msgBuf = nullptr )
  {
    static const int UP = 0;
    static const int LEFT = 1;
    static const int DOWN = 2;
    static const int RIGHT = 3;
    const int inc[4] = {-width,-1,width,1};
    const int incK[4] = {-width*K,-K,width*K,K};
    const int incDim[4] = {-width*K*dim,-K*dim,width*K*dim,K*dim};
    const int begin[4] = {height-1,width-1,0,0};
    const int end[4] = {0,0,height-1,width-1};

    

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

    for ( int iter=0; iter<maxIter; iter++ ) {
      for ( int dir=0; dir<4; dir++ ) {
        int scanBegin = begin[3-dir];
        int scanEnd = end[3-dir];
        for ( int scan=scanBegin; scan<scanEnd; scan++ ) { // outer loop
          for ( int i=begin[dir]; i<end[dir]; i++ ) {
            /*
             * Message:
             * i = pixel index
             * j = pixel index (neighbor)
             * f_i = feature of i-th pixel
             * q_{i,p} = feature of p-th candidates for i-th pixel
             * l_{i,p } = labels of p-th candidates for i-th pixel
             * N(i) = neighbor of i
             * m_{i->j}(q) = min_{p} lambda * ||l_{j,q}-l_{i,p}|| + ||q_{i,p}-f_i||^2 + message(N(i)/j)
             */
            
          }
        }
      }
    }
    
    
    
    
    if ( nullptr == msgBuf ) {
      // free the internally created buffer
      DeleteToNullWithTestArray( buf );
    }
  }
};
