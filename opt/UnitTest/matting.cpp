/*********************************************************************************
 * File: matting.cpp
 * Description: UnitTest for Distance Transform Loopy Belief Propagation
 * by BreakDS, @ University of Wisconsin-Madison, Fri Aug 17 17:58:10 CDT 2012
 *********************************************************************************/
#include <cstdio>
#include "../BP.hpp"
#include "../BP_CUDA.h"
#include "opencv2/opencv.hpp"




void display( uchar *img, int h, int w )
{
  cv::Mat canvas( h, w, CV_8UC1 );
  for ( int i=0; i<h; i++ ) {
    for ( int j=0; j<w; j++ ) {
      canvas.at<uchar>( i, j ) = img[ i * w + j ];
    }
  }
  cv::imshow( "canvas", canvas );
  cv::waitKey();
}

template <typename floating=float>
class RandProj : public optimize::AbstractRandHash<floating>
{
public:
  vector<floating> coeff;
  void shuffle( int dim ) 
  {
    static const float ub = static_cast<double>( RAND_MAX );
    coeff.resize(dim);
    for ( int i=0; i<dim; i++ ) {
      coeff[i] = rand() / ub;
    }
  }
  
  floating operator()( const floating* a, int dim )
  {
    floating sum = 0.01;
    for ( int i=0; i<dim; i++ ) {
      sum += a[i] * coeff[i];
    }
    return sum;
  }
};


int main()
{
  // Generate An M x N image
  int M = 80;
  int N = 80;
  uchar img[N*M];

  

  for ( int i=0; i<M; i++ ) {
    for ( int j=0; j<N; j++ ) {
      float st = M * sin( 3.14 / N * j ) * 0.7;
      if ( static_cast<float>(i) < st ) {
        img[i*N+j] = rand() % 40 + 210;
      } else {
        img[i*N+j] = rand() % 50;
      }
    }
  }


  display( img, M, N );

  // Preparing the candidate
  int K = 4;
  uchar feature[N*M*K];
  float labels[N*M*K*1];
  for ( int i=0; i<N*M; i++ ) {
    labels[i*K] = rand() % 200 + 50;
    labels[i*K+1] = rand() % 20;
    labels[i*K+2] = labels[i*K];
    labels[i*K+3] = labels[i*K+1];
    for ( int k=0; k<K; k++ ) {
      feature[i*K+k] = static_cast<uchar>( labels[i*K+k] );
    }
  }
  
  optimize_cuda::Options options;
  options.maxIter = 10;
  // options.numHypo = 3;
  options.lambda = 1.0;

  float D[N*N*K];
  for ( int i=0; i<N*M; i++ ) {
    for ( int k=0; k<K; k++ ) {
      D[i*K+k] = (feature[i*K+k] - img[i]) * (feature[i*K+k] - img[i] );
    }
  }
  

  int result[M*N];
  
  // optimize::LoopyBP<optimize::FDT<float>,float>( D, labels, 1.0, M, N, K, 1, result, options );
  optimize_cuda::LoopyBP_CUDA_float( D, labels, M, N, K, 1, result, options );

  uchar labeled[M*N];
  for ( int i=0; i<M*N; i++ ) {
    if ( 0 == ( result[i] & 1 ) ) {
      labeled[i] = 255;
    } else {
      labeled[i] = 0;
    }
  }
  
  display( labeled, M, N );

  return 0;
}


